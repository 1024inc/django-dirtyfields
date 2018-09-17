# Adapted from http://stackoverflow.com/questions/110803/dirty-fields-in-django
from copy import deepcopy
import logging

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import DEFERRED
from django.db.models.query import ModelIterable, RelatedPopulator
from django.db.models.expressions import BaseExpression
from django.db.models.expressions import Combinable
from django.db.models.signals import post_save, m2m_changed

from .compare import raw_compare, compare_states, normalise_value
from .compat import is_buffer, get_m2m_with_model, remote_field
from .decorators import require_non_disabled


log = logging.getLogger(__name__)


# NOTE: Overridden to pass the flag to disable dirtyfields for the select related objects
# Additions are surrounded by ### comment blocks
class _RelatedPopulator(RelatedPopulator):

    def populate(self, row, from_obj):

        ###
        # Get the flag from the object
        dirtyfields_disabled = from_obj._dirtyfields_disabled
        ###

        if self.reorder_for_init:
            obj_data = self.reorder_for_init(row)
        else:
            obj_data = row[self.cols_start:self.cols_end]
        if obj_data[self.pk_idx] is None:
            obj = None
        else:
            ###
            # Pass disable_dirtyfield only if it's a subclass of DirtyFieldsMixin
            # because only a subclass of DirtyFieldsMixin will support this kwarg
            if issubclass(self.model_cls, DirtyFieldsMixin):
                obj = self.model_cls.from_db(
                    self.db, self.init_list, obj_data,
                    disable_dirtyfields=dirtyfields_disabled
                )
            else:
                if dirtyfields_disabled:
                    log.warning(
                        f'Dirtyfields is disabled but this model does not '
                        f'support it: {self.model_cls} '
                        f'=> Inherit from DirtyFieldsMixin to support it'
                    )
                obj = self.model_cls.from_db(self.db, self.init_list, obj_data)
            ###

        if obj and self.related_populators:
            for rel_iter in self.related_populators:
                rel_iter.populate(row, obj)
        setattr(from_obj, self.cache_name, obj)
        if obj and self.reverse_cache_name:
            setattr(obj, self.reverse_cache_name, from_obj)


# NOTE: Overridden to use our RelatedPopulator
def get_related_populators(klass_info, select, db):
    iterators = []
    related_klass_infos = klass_info.get('related_klass_infos', [])
    for rel_klass_info in related_klass_infos:
        ###
        # Use our RelatedPopulator
        rel_cls = _RelatedPopulator(rel_klass_info, select, db)
        ###
        iterators.append(rel_cls)
    return iterators


# NOTE: Overridden to pass the flag to disable dirtyfields from the queryset to the model creation
# Additions are surrounded by ### comment blocks
class _ModelIterable(ModelIterable):

    def __iter__(self):
        queryset = self.queryset

        ###
        # Get disable_dirtyfields from the queryset
        disable_dirtyfields = queryset._disable_dirtyfields
        ###

        db = queryset.db
        compiler = queryset.query.get_compiler(using=db)
        # Execute the query. This will also fill compiler.select, klass_info,
        # and annotations.
        results = compiler.execute_sql(chunked_fetch=self.chunked_fetch)
        select, klass_info, annotation_col_map = (compiler.select, compiler.klass_info,
                                                  compiler.annotation_col_map)
        model_cls = klass_info['model']
        select_fields = klass_info['select_fields']
        model_fields_start, model_fields_end = select_fields[0], select_fields[-1] + 1
        init_list = [f[0].target.attname
                     for f in select[model_fields_start:model_fields_end]]
        related_populators = get_related_populators(klass_info, select, db)
        for row in compiler.results_iter(results):

            ###
            # Pass disable_dirtyfield only if it's a subclass of DirtyFieldsMixin
            # because only a subclass of DirtyFieldsMixin will support this kwarg
            if issubclass(model_cls, DirtyFieldsMixin):
                obj = model_cls.from_db(
                    db, init_list, row[model_fields_start:model_fields_end],
                    ###
                    # Pass disable_dirtyfields
                    disable_dirtyfields=disable_dirtyfields
                    ###
                )
            else:
                if disable_dirtyfields:
                    log.warning(
                        f'Dirtyfields is disabled but this model does not '
                        f'support it: {model_cls} '
                        f'=> Inherit from DirtyFieldsMixin to support it'
                    )
                obj = model_cls.from_db(db, init_list, row[model_fields_start:model_fields_end])
            ###

            if related_populators:
                for rel_populator in related_populators:
                    rel_populator.populate(row, obj)
            if annotation_col_map:
                for attr_name, col_pos in annotation_col_map.items():
                    setattr(obj, attr_name, row[col_pos])

            # Add the known related objects to the model, if there are any
            if queryset._known_related_objects:
                for field, rel_objs in queryset._known_related_objects.items():
                    # Avoid overwriting objects loaded e.g. by select_related
                    if hasattr(obj, field.get_cache_name()):
                        continue
                    pk = getattr(obj, field.get_attname())
                    try:
                        rel_obj = rel_objs[pk]
                    except KeyError:
                        pass  # may happen in qs1 | qs2 scenarios
                    else:
                        setattr(obj, field.name, rel_obj)

            yield obj


class DirtyFieldsQuerySet(models.QuerySet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace with our ModelIterable
        self._iterable_class = _ModelIterable

        # Flag to know whether dirtyfields needs to be disabled or not
        # Not disabled by default
        self._disable_dirtyfields = False

    def disable_dirtyfields(self):
        """Returns a new queryset with dirtyfields disabled"""
        clone = self._clone()
        # Set to True
        clone._disable_dirtyfields = True
        return clone

    # NOTE: Overridden to pass disable_dirtyfields to the clone
    def _clone(self, **kwargs):
        clone = super()._clone(**kwargs)

        # Pass disable_dirtyfields to the cloned queryset
        clone._disable_dirtyfields = self._disable_dirtyfields

        return clone


class DirtyFieldsMixin(object):
    compare_function = (raw_compare, {})
    normalise_function = (normalise_value, {})

    # This mode has been introduced to handle some situations like this one:
    # https://github.com/romgar/django-dirtyfields/issues/73
    ENABLE_M2M_CHECK = False

    FIELDS_TO_CHECK = None

    def __init__(self, *args, **kwargs):

        # Get the flag to know whether to disable dirtyfields
        self._dirtyfields_disabled = kwargs.pop('disable_dirtyfields', False)

        log.debug(f'__init__ {self._dirtyfields_disabled}')

        super(DirtyFieldsMixin, self).__init__(*args, **kwargs)

        # Init state only if not disabled
        if not self._dirtyfields_disabled:
            post_save.connect(
                reset_state, sender=self.__class__, weak=False,
                dispatch_uid='{name}-DirtyFieldsMixin-sweeper'.format(
                    name=self.__class__.__name__))
            if self.ENABLE_M2M_CHECK:
                self._connect_m2m_relations()
            reset_state(sender=self.__class__, instance=self)

    # NOTE: Overridden to pass disable_dirtyfields
    # Additions are surrounded by ### comment blocks
    @classmethod
    def from_db(cls, db, field_names, values, disable_dirtyfields=None):
        if len(values) != len(cls._meta.concrete_fields):
            values = list(values)
            values.reverse()
            values = [values.pop() if f.attname in field_names else DEFERRED for f in cls._meta.concrete_fields]

        ###
        # Pass disable_dirtyfields to the model
        log.debug(f'from_db {disable_dirtyfields} - {cls}')
        new = cls(*values, disable_dirtyfields=disable_dirtyfields)
        ###

        new._state.adding = False
        new._state.db = db
        return new

    def refresh_from_db(self, *a, **kw):
        super(DirtyFieldsMixin, self).refresh_from_db(*a, **kw)

        # Reset state only if not disabled
        if not self._dirtyfields_disabled:
            reset_state(sender=self.__class__, instance=self)

    @require_non_disabled
    def _connect_m2m_relations(self):
        for m2m_field, model in get_m2m_with_model(self.__class__):
            m2m_changed.connect(
                reset_state, sender=remote_field(m2m_field).through, weak=False,
                dispatch_uid='{name}-DirtyFieldsMixin-sweeper-m2m'.format(
                    name=self.__class__.__name__))

    @require_non_disabled
    def _as_dict(self, check_relationship, include_primary_key=True):
        all_field = {}

        deferred_fields = self.get_deferred_fields()

        for field in self._meta.fields:
            if self.FIELDS_TO_CHECK and (field.get_attname() not in self.FIELDS_TO_CHECK):
                continue

            if field.primary_key and not include_primary_key:
                continue

            if remote_field(field):
                if not check_relationship:
                    continue

            if field.get_attname() in deferred_fields:
                continue

            field_value = getattr(self, field.attname)

            # If current field value is an expression, we are not evaluating it
            if isinstance(field_value, (BaseExpression, Combinable)):
                continue

            try:
                # Store the converted value for fields with conversion
                field_value = field.to_python(field_value)
            except ValidationError:
                # The current value is not valid so we cannot convert it
                pass

            if is_buffer(field_value):
                # psycopg2 returns uncopyable type buffer for bytea
                field_value = bytes(field_value)

            # Use the column name (instead of the relationship name) if it's a
            # foreign key.
            key = field.attname if hasattr(field, 'attname') else field.name
            # Explanation of copy usage here :
            # https://github.com/romgar/django-dirtyfields/commit/efd0286db8b874b5d6bd06c9e903b1a0c9cc6b00
            all_field[key] = deepcopy(field_value)

        return all_field

    @require_non_disabled
    def _as_dict_m2m(self):
        m2m_fields = {}

        if self.pk:
            for f, model in get_m2m_with_model(self.__class__):
                if self.FIELDS_TO_CHECK and (f.attname not in self.FIELDS_TO_CHECK):
                    continue

                m2m_fields[f.attname] = set([obj.pk for obj in getattr(self, f.attname).all()])

        return m2m_fields

    @require_non_disabled
    def get_dirty_fields(self, check_relationship=False, check_m2m=None, verbose=False):
        if self._state.adding:
            # If the object has not yet been saved in the database, all fields are considered dirty
            # for consistency (see https://github.com/romgar/django-dirtyfields/issues/65 for more details)
            pk_specified = self.pk is not None
            initial_dict = self._as_dict(check_relationship, include_primary_key=pk_specified)
            if verbose:
                initial_dict = {key: {'saved': None, 'current': self.normalise_function[0](value)}
                                for key, value in initial_dict.items()}
            return initial_dict

        if check_m2m is not None and not self.ENABLE_M2M_CHECK:
            raise ValueError("You can't check m2m fields if ENABLE_M2M_CHECK is set to False")

        modified_fields = compare_states(self._as_dict(check_relationship),
                                         self._original_state,
                                         self.compare_function,
                                         self.normalise_function)

        if check_m2m:
            modified_m2m_fields = compare_states(check_m2m,
                                                 self._original_m2m_state,
                                                 self.compare_function,
                                                 self.normalise_function)
            modified_fields.update(modified_m2m_fields)

        if not verbose:
            # Keeps backward compatibility with previous function return
            modified_fields = {key: self.normalise_function[0](value['saved']) for key, value in modified_fields.items()}

        return modified_fields

    @require_non_disabled
    def is_dirty(self, check_relationship=False, check_m2m=None):
        return {} != self.get_dirty_fields(check_relationship=check_relationship,
                                           check_m2m=check_m2m)

    @require_non_disabled
    def save_dirty_fields(self):
        dirty_fields = self.get_dirty_fields(check_relationship=True)
        self.save(update_fields=dirty_fields.keys())


def reset_state(sender, instance, **kwargs):
    log.debug(f'RESET STATE: {sender} {instance}')
    # original state should hold all possible dirty fields to avoid
    # getting a `KeyError` when checking if a field is dirty or not
    update_fields = kwargs.pop('update_fields', {})
    new_state = instance._as_dict(check_relationship=True)
    FIELDS_TO_CHECK = getattr(instance, "FIELDS_TO_CHECK", None)
    if update_fields:
        for field_name in update_fields:
            field = sender._meta.get_field(field_name)
            if not FIELDS_TO_CHECK or \
                    (field.get_attname() in FIELDS_TO_CHECK):

                if field.get_attname() in instance.get_deferred_fields():
                    continue

                # Use the column name (instead of the relationship name) if it's a
                # foreign key.
                key = field.attname if hasattr(field, 'attname') else field.name
                instance._original_state[key] = new_state[key]

    else:
        instance._original_state = new_state
    if instance.ENABLE_M2M_CHECK:
        instance._original_m2m_state = instance._as_dict_m2m()

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

from marshmallow import fields
from marshmallow.utils import is_iterable_but_not_string, missing
from sqlalchemy import inspect
from sqlalchemy.orm.exc import NoResultFound

if TYPE_CHECKING:
    from sqlalchemy.ext.declarative import DeclarativeMeta
    from sqlalchemy.orm import MapperProperty

from .loaded_only import OMITTED


class RelatedList(fields.List):
    def get_value(self, obj, attr, accessor=None):
        # Do not call `fields.List`'s get_value as it calls the container's
        # `get_value` if the container has `attribute`.
        # Instead call the `get_value` from the parent of `fields.List`
        # so the special handling is avoided.
        return super(fields.List, self).get_value(obj, attr, accessor=accessor)


class Related(fields.Field):
    """Related data represented by a SQLAlchemy `relationship`. Must be attached
    to a `Schema <marshmallow.Schema>` class whose options includes a SQLAlchemy `model`, such
    as `SQLAlchemySchema <marshmallow_sqlalchemy.SQLAlchemySchema>`.

    :param columns: Optional column names on related model. If not provided,
        the primary key(s) of the related model will be used.
    """

    default_error_messages = {
        "invalid": "Could not deserialize related value {value!r}; "
        "expected a dictionary with keys {keys!r}"
    }

    def __init__(
        self,
        columns: list[str] | str | None = None,
        column: str | None = None,
        **kwargs,
    ):
        if column is not None:
            warnings.warn(
                "`column` parameter is deprecated and will be removed in future releases. "
                "Use `columns` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if columns is None:
                columns = column
        super().__init__(**kwargs)
        self.columns: list[str] = ensure_list(columns or [])

    @property
    def model(self) -> type[DeclarativeMeta] | None:
        if self.root is None:
            raise RuntimeError("Cannot access model before field is bound to schema.")
        return self.root.opts.model

    @property
    def related_model(self) -> type[DeclarativeMeta]:
        if self.model is None:
            raise RuntimeError(
                "Cannot access related_model if schema does not have a model."
            )
        model_attr = getattr(self.model, cast("str", self.attribute or self.name))
        if hasattr(model_attr, "remote_attr"):  # handle association proxies
            model_attr = model_attr.remote_attr
        return model_attr.property.mapper.class_

    @property
    def related_keys(self):
        if self.columns:
            insp = inspect(self.related_model)
            return [insp.attrs[column] for column in self.columns]
        return get_primary_keys(self.related_model)

    @property
    def session(self):
        return self.root.session

    @property
    def transient(self):
        return self.root.transient

    def _serialize(self, value, attr, obj):
        from .loaded_only import current_strategy, get_loaded_value

        active = current_strategy() is not None
        ret = {}
        for prop in self.related_keys:
            if active:
                # The outer field name (e.g. "school") is already on the path.
                # For composite keys the related column name is appended so the
                # message points at e.g. "school.name".
                composite = len(self.related_keys) > 1
                key_value = get_loaded_value(
                    value, prop.key, prop.key if composite else None
                )
                if key_value is OMITTED:
                    return missing
            else:
                key_value = getattr(value, prop.key, None)
            ret[prop.key] = key_value
        return ret if len(ret) > 1 else next(iter(ret.values()))

    def _deserialize(self, value, *args, **kwargs):
        """Deserialize a serialized value to a model instance.

        If the parent schema is transient, create a new (transient) instance.
        Otherwise, attempt to find an existing instance in the database.
        :param value: The value to deserialize.
        """
        if not isinstance(value, dict):
            if len(self.related_keys) != 1:
                keys = [prop.key for prop in self.related_keys]
                raise self.make_error("invalid", value=value, keys=keys)
            value = {self.related_keys[0].key: value}
        if self.transient:
            return self.related_model(**value)
        try:
            result = self._get_existing_instance(self.related_model, value)
        except NoResultFound:
            # The related-object DNE in the DB, but we still want to deserialize it
            # ...perhaps we want to add it to the DB later
            return self.related_model(**value)
        return result

    def _get_existing_instance(self, related_model, value):
        """Retrieve the related object from an existing instance in the DB.

        :param related_model: The related model to query
        :param value: The serialized value to mapto an existing instance.
        :raises NoResultFound: if there is no matching record.
        """
        if self.columns:
            result = (
                self.session.query(related_model)
                .filter_by(
                    **{prop.key: value.get(prop.key) for prop in self.related_keys}
                )
                .one()
            )
        else:
            # Use a faster path if the related key is the primary key.
            lookup_values = [value.get(prop.key) for prop in self.related_keys]
            try:
                result = self.session.get(related_model, lookup_values)
            except TypeError as error:
                keys = [prop.key for prop in self.related_keys]
                raise self.make_error("invalid", value=value, keys=keys) from error
            if result is None:
                raise NoResultFound
        return result


class Nested(fields.Nested):
    """Nested field that inherits the session from its parent."""

    def _serialize(self, nested_obj, attr, obj, **kwargs):
        from contextlib import ExitStack

        from .loaded_only import current_path, current_strategy, path_segment

        nested_schema = self.schema
        # The nested serialization is loaded-only when an ancestor already
        # activated the mode or the nested schema itself enables it.
        child_mode = getattr(nested_schema, "loaded_only", False)
        if current_strategy() is None and not child_mode:
            return super()._serialize(nested_obj, attr, obj, **kwargs)
        with ExitStack() as stack:
            # The field name is already on the path inside a SQLAlchemySchema;
            # a plain marshmallow parent (including a schema enabling the mode
            # only on the child) needs us to push it here (idempotently).
            parent_owns_name = (
                current_strategy() is not None
                and current_path()
                and current_path()[-1] == self.name
            )
            if not parent_owns_name:
                stack.enter_context(path_segment(self.name))
            index = self._index_in_parent(nested_obj, attr, obj)
            if index is not None:
                stack.enter_context(path_segment(index))
            return super()._serialize(nested_obj, attr, obj, **kwargs)

    def _index_in_parent(self, nested_obj, attr, obj):
        """Return the index of ``nested_obj`` for ``fields.List(Nested(...))``.

        Returns ``None`` for single nested objects. The parent collection is
        already loaded at this point (the surrounding List field read it), so
        identity comparison never emits SQL.
        """
        inner = getattr(self.parent, "inner", None)
        if inner is not self:
            return None
        try:
            container = getattr(obj, attr)
        except AttributeError:
            return None
        if is_iterable_but_not_string(container):
            for index, item in enumerate(container):
                if item is nested_obj:
                    return index
        return None

    def _deserialize(self, *args, **kwargs):
        if hasattr(self.schema, "session"):
            self.schema.session = self.root.session
            self.schema.transient = self.root.transient
        return super()._deserialize(*args, **kwargs)


def get_primary_keys(model: type[DeclarativeMeta]) -> list[MapperProperty]:
    """Get primary key properties for a SQLAlchemy model.

    :param model: SQLAlchemy model class
    """
    mapper = model.__mapper__  # type: ignore[attr-defined]
    return [mapper.get_property_by_column(column) for column in mapper.primary_key]


def ensure_list(value: Any) -> list:
    return list(value) if is_iterable_but_not_string(value) else [value]

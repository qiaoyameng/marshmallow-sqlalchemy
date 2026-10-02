from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

from marshmallow import fields
from marshmallow.utils import is_iterable_but_not_string
from sqlalchemy import inspect
from sqlalchemy.orm.exc import NoResultFound

if TYPE_CHECKING:
    from sqlalchemy.ext.declarative import DeclarativeMeta
    from sqlalchemy.orm import MapperProperty


class RelatedList(fields.List):
    def __init__(self, cls_or_instance, **kwargs):
        # `require_existing` applies to the inner `Related` field, not the
        # list itself, so it must not be forwarded to `fields.List`.
        require_existing = kwargs.pop("require_existing", None)
        super().__init__(cls_or_instance, **kwargs)
        if require_existing is not None and isinstance(self.inner, Related):
            self.inner.require_existing = require_existing

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
    :param require_existing: If `True`, deserialization only accepts values
        that identify an existing related object. A validation error is raised
        when the key is incomplete, when no matching row exists, or when no
        session is available to perform the lookup. Objects that are pending
        in the session (registered but not yet committed) are valid targets.
        The lookup never triggers an autoflush. Defaults to `False`, which
        preserves the historical behavior of creating a new (transient)
        instance when no existing object is found.
    """

    default_error_messages = {
        "invalid": "Could not deserialize related value {value!r}; "
        "expected a dictionary with keys {keys!r}",
        "does_not_exist": "Could not find existing related object "
        "with keys {keys!r} matching {value!r}",
        "incomplete_key": "Could not deserialize related value {value!r}; "
        "missing values for keys {keys!r}",
        "no_session": "Could not deserialize related value {value!r}; "
        "an existing object is required but no session is available",
    }

    def __init__(
        self,
        columns: list[str] | str | None = None,
        column: str | None = None,
        require_existing: bool = False,
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
        self.require_existing = require_existing

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
        ret = {prop.key: getattr(value, prop.key, None) for prop in self.related_keys}
        return ret if len(ret) > 1 else next(iter(ret.values()))

    def _deserialize(self, value, *args, **kwargs):
        """Deserialize a serialized value to a model instance.

        If the parent schema is transient, create a new (transient) instance.
        Otherwise, attempt to find an existing instance in the database.
        If ``require_existing`` is `True`, only existing instances are
        returned and a validation error is raised otherwise.
        :param value: The value to deserialize.
        """
        if not isinstance(value, dict):
            if len(self.related_keys) != 1:
                keys = [prop.key for prop in self.related_keys]
                raise self.make_error("invalid", value=value, keys=keys)
            value = {self.related_keys[0].key: value}
        if self.require_existing:
            return self._get_required_instance(self.related_model, value)
        if self.transient:
            return self.related_model(**value)
        try:
            result = self._get_existing_instance(self.related_model, value)
        except NoResultFound:
            # The related-object DNE in the DB, but we still want to deserialize it
            # ...perhaps we want to add it to the DB later
            return self.related_model(**value)
        return result

    def _get_required_instance(self, related_model, value):
        """Retrieve an existing related object, never creating a new one.

        :param related_model: The related model to query
        :param value: The serialized value to map to an existing instance.
        :raises ValidationError: if the key is incomplete, if no session is
            available, or if there is no matching record.
        """
        keys = [prop.key for prop in self.related_keys]
        if self.transient or self.session is None:
            raise self.make_error("no_session", value=value, keys=keys)
        missing_keys = [key for key in keys if value.get(key) is None]
        if missing_keys:
            raise self.make_error("incomplete_key", value=value, keys=missing_keys)
        # Pending (unflushed) objects registered in the session are valid
        # targets; match them in memory before hitting the database.
        result = self._get_pending_by_keys(related_model, value)
        if result is not None:
            return result
        if self.columns:
            result = self._get_existing_by_columns(related_model, value)
        else:
            # `no_autoflush` guarantees the lookup never flushes unrelated
            # pending changes to the database.
            lookup_values = [value.get(prop.key) for prop in self.related_keys]
            try:
                with self.session.no_autoflush:
                    result = self.session.get(related_model, lookup_values)
            except TypeError as error:
                raise self.make_error("invalid", value=value, keys=keys) from error
        if result is None:
            raise self.make_error("does_not_exist", value=value, keys=keys)
        return result

    def _get_pending_by_keys(self, related_model, value):
        """Find a matching object among the pending objects of the session."""
        for obj in self.session.new:
            if isinstance(obj, related_model) and all(
                getattr(obj, prop.key, None) == value.get(prop.key)
                for prop in self.related_keys
            ):
                return obj
        return None

    def _get_existing_by_columns(self, related_model, value):
        """Find an existing object by non-primary-key columns.

        The database query runs under ``no_autoflush`` so that
        deserialization never writes unrelated changes to the database.
        """
        with self.session.no_autoflush:
            try:
                return (
                    self.session.query(related_model)
                    .filter_by(
                        **{
                            prop.key: value.get(prop.key)
                            for prop in self.related_keys
                        }
                    )
                    .one()
                )
            except NoResultFound:
                return None

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

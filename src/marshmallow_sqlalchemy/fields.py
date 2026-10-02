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
        # `require_existing` is handled by the inner `Related` field, which
        # receives it through the model converter. It is not a valid
        # `fields.List` keyword argument, so it is dropped here.
        kwargs.pop("require_existing", None)
        super().__init__(cls_or_instance, **kwargs)

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
    :param require_existing: If `True`, deserialization follows an
        "existing-only" strategy: the related value must resolve to a record
        that already exists, and a validation error is raised otherwise.
        Objects registered in the session but not yet committed
        (``session.new``) are considered existing and may be referenced.
        The lookup never triggers an autoflush, so deserialization cannot
        cause unrelated writes to the database. If `False` (the default),
        a new transient instance is created when no matching record exists,
        preserving the historical behavior. This option has no effect when
        the schema is transient, since transient loading never uses the
        session.
    """

    default_error_messages = {
        "invalid": "Could not deserialize related value {value!r}; "
        "expected a dictionary with keys {keys!r}",
        "not_found": "Could not find related object with keys {keys!r} "
        "for value {value!r}",
        "incomplete_key": "Missing related key(s) {missing!r}; "
        "expected a dictionary with keys {keys!r}",
        "no_session": "Could not resolve related object without a "
        "SQLAlchemy session",
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
        :param value: The value to deserialize.
        """
        if not isinstance(value, dict):
            if len(self.related_keys) != 1:
                keys = [prop.key for prop in self.related_keys]
                raise self.make_error("invalid", value=value, keys=keys)
            value = {self.related_keys[0].key: value}
        if self.transient:
            return self.related_model(**value)
        if self.require_existing:
            return self._get_required_instance(value)
        try:
            result = self._get_existing_instance(self.related_model, value)
        except NoResultFound:
            # The related-object DNE in the DB, but we still want to deserialize it
            # ...perhaps we want to add it to the DB later
            return self.related_model(**value)
        return result

    def _get_required_instance(self, value):
        """Retrieve an existing related object, or raise a validation error.

        Only existing records are bound: objects registered in the session
        but not yet committed (``session.new``) and records persisted in the
        database are both valid targets. No new instance is ever created,
        and the database lookup is performed with autoflush disabled so that
        deserialization cannot trigger unrelated writes.

        :param value: The serialized value to map to an existing instance.
        :raises ValidationError: if no session is available, if the key is
            incomplete, or if no matching record exists.
        """
        keys = [prop.key for prop in self.related_keys]
        if self.session is None:
            raise self.make_error("no_session")
        missing = [key for key in keys if value.get(key) is None]
        if missing:
            raise self.make_error("incomplete_key", missing=missing, keys=keys)
        pending = self._get_pending_instance(value)
        if pending is not None:
            return pending
        # `no_autoflush` guarantees the lookup cannot flush unrelated
        # pending changes to the database as a side effect of deserializing.
        with self.session.no_autoflush:
            try:
                return self._get_existing_instance(self.related_model, value)
            except NoResultFound:
                raise self.make_error("not_found", value=value, keys=keys) from None

    def _get_pending_instance(self, value):
        """Return a matching object registered in the session but not yet
        committed, or `None` if there is no match.

        Pending objects are valid deserialization targets: they are already
        registered in the session and will be persisted on the next flush.
        """
        for obj in self.session.new:
            if isinstance(obj, self.related_model) and all(
                getattr(obj, prop.key, None) == value.get(prop.key)
                for prop in self.related_keys
            ):
                return obj
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

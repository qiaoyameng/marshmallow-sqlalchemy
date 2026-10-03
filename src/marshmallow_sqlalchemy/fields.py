from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

from marshmallow import ValidationError, fields
from marshmallow.utils import is_iterable_but_not_string
from sqlalchemy import inspect
from sqlalchemy.orm.exc import NoResultFound

if TYPE_CHECKING:
    from sqlalchemy.ext.declarative import DeclarativeMeta
    from sqlalchemy.orm import MapperProperty


class RelatedList(fields.List):
    def get_value(self, obj, attr, accessor=None):
        # Do not call `fields.List`'s get_value as it calls the container's
        # `get_value` if the container has `attribute`.
        # Instead call the `get_value` from the parent of `fields.List`
        # so the special handling is avoided.
        return super(fields.List, self).get_value(obj, attr, accessor=accessor)

    def _deserialize(self, value, attr, data, **kwargs):
        if (
            isinstance(value, list)
            and getattr(self.root, "_load_instance", False)
            and getattr(self.root, "_load_phase", None) == "validate"
        ):
            targets = set()
            for index, item in enumerate(value):
                raw_value = item
                try:
                    related_keys = self.inner.related_keys
                    if not isinstance(raw_value, dict):
                        if len(related_keys) != 1:
                            continue
                        raw_value = {related_keys[0].key: item}
                    key = tuple(
                        raw_value.get(prop.key) for prop in related_keys
                    )
                    if None in key:
                        continue
                    if key in targets:
                        raise ValidationError(
                            {
                                index: [
                                    "Duplicate target for "
                                    f"{self.inner.related_model.__name__}: "
                                    + ",".join(str(part) for part in key)
                                ]
                            }
                        )
                    targets.add(key)
                except RuntimeError:
                    return super()._deserialize(value, attr, data, **kwargs)
        return super()._deserialize(value, attr, data, **kwargs)


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
            lookup_values = tuple(
                value.get(prop.key) for prop in self.related_keys
            )
            lookup_values = (
                next(iter(lookup_values))
                if len(lookup_values) == 1
                else lookup_values
            )
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

    def _load(self, value, partial=None):
        schema = self.schema
        if hasattr(schema, "session"):
            schema.session = self.root.session
            schema.transient = self.root.transient
        phase = (
            getattr(self.root, "_load_phase", None)
            if hasattr(schema, "_load_phase")
            and getattr(schema, "_load_instance", False)
            else None
        )
        previous_two_phase = getattr(schema, "_two_phase", False)
        if phase is not None:
            schema._load_phase = phase
            schema._two_phase = True
        try:
            if phase == "validate":
                try:
                    return schema._do_load(
                        value,
                        many=schema.many,
                        partial=partial,
                        unknown=self.unknown,
                        postprocess=True,
                    )
                except ValidationError as error:
                    raise ValidationError(
                        error.messages, valid_data=error.valid_data
                    ) from error
            return super()._load(value, partial=partial)
        finally:
            if phase is not None:
                schema._load_phase = None
                schema._two_phase = previous_two_phase

    def _deserialize(self, *args, **kwargs):
        return super()._deserialize(*args, **kwargs)


def get_primary_keys(model: type[DeclarativeMeta]) -> list[MapperProperty]:
    """Get primary key properties for a SQLAlchemy model.

    :param model: SQLAlchemy model class
    """
    mapper = model.__mapper__  # type: ignore[attr-defined]
    return [mapper.get_property_by_column(column) for column in mapper.primary_key]


def ensure_list(value: Any) -> list:
    return list(value) if is_iterable_but_not_string(value) else [value]

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping
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


class Composite(Nested):
    """Field for a SQLAlchemy composite property."""

    def __init__(
        self,
        nested,
        composite_class,
        component_keys: Iterable[str],
        *,
        construct=None,
        value_getter: Callable[[Any], Iterable[Any]] | None = None,
        **kwargs,
    ):
        super().__init__(nested, **kwargs)
        self.composite_class = composite_class
        self.component_keys = tuple(component_keys)
        self.construct = construct or composite_class
        self.value_getter = value_getter

    def _load(
        self,
        value,
        partial: bool | str | list | tuple | set | None = None,
    ):
        if value is None:
            return None

        if partial is not False:
            if isinstance(value, Mapping):
                current = getattr(self.root, "instance", None)
                if current is not None:
                    current = getattr(current, self.attribute or self.name, None)

                values = {}
                for key in self.component_keys:
                    field = self.schema.fields.get(key)
                    if field is None:
                        continue
                    data_key = field.data_key or key
                    if data_key in value:
                        values[data_key] = value[data_key]
                    elif current is not None:
                        values[data_key] = self._get_component_value(current, key)

                value = values
            partial = False

        try:
            data = super()._load(value, partial=partial)
            values = []
            for key in self.component_keys:
                field = self.schema.fields.get(key)
                data_key = field.attribute or key if field else key
                values.append(data.get(data_key))
            return self.construct(*values)
        except ValidationError:
            raise
        except (TypeError, ValueError) as error:
            raise ValidationError(str(error), field_name="_schema") from error

    def get_component_values(self, value):
        if self.value_getter is not None:
            try:
                values = tuple(self.value_getter(value))
            except AttributeError:
                values = ()
            if len(values) == len(self.component_keys):
                return values
        try:
            return tuple(value.__composite_values__())
        except AttributeError:
            return tuple(getattr(value, key) for key in self.component_keys)

    def _get_component_value(self, value, key):
        values = self.get_component_values(value)
        if len(values) == len(self.component_keys):
            return values[self.component_keys.index(key)]
        return getattr(value, key)


def get_primary_keys(model: type[DeclarativeMeta]) -> list[MapperProperty]:
    """Get primary key properties for a SQLAlchemy model.

    :param model: SQLAlchemy model class
    """
    mapper = model.__mapper__  # type: ignore[attr-defined]
    return [mapper.get_property_by_column(column) for column in mapper.primary_key]


def ensure_list(value: Any) -> list:
    return list(value) if is_iterable_but_not_string(value) else [value]

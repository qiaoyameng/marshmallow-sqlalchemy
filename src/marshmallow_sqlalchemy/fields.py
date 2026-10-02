from __future__ import annotations

from collections.abc import Mapping, Sequence
import warnings
from typing import TYPE_CHECKING, Any, cast

import sqlalchemy as sa
from marshmallow import ValidationError, fields
from marshmallow.utils import is_iterable_but_not_string
from sqlalchemy import inspect
from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound

if TYPE_CHECKING:
    from sqlalchemy.ext.declarative import DeclarativeMeta
    from sqlalchemy.orm import MapperProperty
    from sqlalchemy.orm import Session


_MISSING = object()


class RelatedBatch:
    def __init__(
        self,
        schema: Any,
        data: Any,
        *,
        session: Session,
        batch_size: int,
    ):
        self.session = session
        self.batch_size = batch_size
        self.groups: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.results: dict[tuple[Any, ...], dict[tuple[Any, ...], Any]] = {}
        self.fallbacks: dict[tuple[Any, ...], Any] = {}
        self._collect(schema, data)
        self._resolve()

    def get(self, field: Related, value: Any) -> Any:
        normalized_value = field.normalize_related_value(value)
        key = field.related_key(normalized_value)
        group_key = self.group_key(field)
        result = self.results.get(group_key, {}).get(key, _MISSING)
        if result is not _MISSING:
            return result

        fallback_key = (id(field), key)
        fallback = self.fallbacks.get(fallback_key, _MISSING)
        if fallback is _MISSING:
            fallback = field.related_model(**normalized_value)
            self.fallbacks[fallback_key] = fallback
        return fallback

    def _collect(self, schema: Any, data: Any) -> None:
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes, Mapping)):
            return

        for item in data:
            if not isinstance(item, Mapping):
                continue
            for field_name, field in schema.load_fields.items():
                related = self._related_field(field)
                if related is None or related.transient:
                    continue

                data_key = related.data_key or field_name
                raw_value = item.get(data_key, _MISSING)
                if raw_value is _MISSING or raw_value is None:
                    continue

                pre_load_fields = (
                    field.inner.pre_load
                    if isinstance(field, RelatedList)
                    else field.pre_load
                )
                for func in pre_load_fields:
                    raw_value = func(raw_value)

                values: Any
                if isinstance(field, RelatedList):
                    values = raw_value
                    if not isinstance(values, Sequence) or isinstance(
                        values, (str, bytes, Mapping)
                    ):
                        continue
                else:
                    if isinstance(raw_value, (str, bytes)):
                        continue
                    values = [raw_value]

                group_key = self.group_key(related)
                group = self.groups.setdefault(
                    group_key,
                    {
                        "field": related,
                        "model": related.related_model,
                        "keys": set(),
                    },
                )
                for value in values:
                    try:
                        if isinstance(field, RelatedList):
                            for func in related.pre_load:
                                value = func(value)
                        normalized_value = related.normalize_related_value(value)
                        key = related.related_key(normalized_value)
                    except ValidationError:
                        continue
                    group["keys"].add(key)

    def _resolve(self) -> None:
        for group_key, group in self.groups.items():
            field = group["field"]
            model = group["model"]
            keys = group["keys"]
            found: dict[tuple[Any, ...], Any] = {}
            keys_to_query = list(keys)

            if not field.columns:
                mapper = inspect(model)
                remaining_keys = []
                for key in keys:
                    identity_key = mapper.identity_key_from_primary_key(key)
                    instance = self.session.identity_map.get(identity_key)
                    if instance is None:
                        remaining_keys.append(key)
                    else:
                        found[key] = instance
                keys_to_query = remaining_keys

            if keys_to_query:
                key_count = len(field.related_keys)
                chunk_size = self.batch_size // key_count
                if chunk_size < 1:
                    raise ValueError(
                        "`related_batch_size` must be at least the number of "
                        "related key columns."
                    )
                for start in range(0, len(keys_to_query), chunk_size):
                    chunk = keys_to_query[start : start + chunk_size]
                    self._query_chunk(field, model, chunk, found)

            self.results[group_key] = found

    def _query_chunk(self, field, model, keys, found) -> None:
        columns = [getattr(model, prop.key) for prop in field.related_keys]
        if len(columns) == 1:
            condition = columns[0].in_([key[0] for key in keys])
        else:
            condition = sa.tuple_(*columns).in_(list(keys))

        instances = self.session.query(model).filter(condition).all()
        chunk_keys = set()
        for instance in instances:
            key = tuple(getattr(instance, prop.key) for prop in field.related_keys)
            if key in chunk_keys:
                raise MultipleResultsFound(
                    f"Multiple rows found for related key {key!r}"
                )
            chunk_keys.add(key)
            found[key] = instance

    @staticmethod
    def _related_field(field: fields.Field) -> Related | None:
        if isinstance(field, Related):
            return field
        if isinstance(field, RelatedList):
            return field.inner
        return None

    @staticmethod
    def group_key(field: Related) -> tuple[Any, ...]:
        return (
            field.related_model,
            tuple(prop.key for prop in field.related_keys),
            bool(field.columns),
        )


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
        value = self.normalize_related_value(value)
        if self.transient:
            return self.related_model(**value)

        batch = getattr(self.root, "_related_batch", None)
        if isinstance(batch, RelatedBatch):
            return batch.get(self, value)

        try:
            result = self._get_existing_instance(self.related_model, value)
        except NoResultFound:
            # The related-object DNE in the DB, but we still want to deserialize it
            # ...perhaps we want to add it to the DB later
            return self.related_model(**value)
        return result

    def normalize_related_value(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            if len(self.related_keys) != 1:
                keys = [prop.key for prop in self.related_keys]
                raise self.make_error("invalid", value=value, keys=keys)
            value = {self.related_keys[0].key: value}
        return value

    def related_key(self, value: dict[str, Any]) -> tuple[Any, ...]:
        key = tuple(value.get(prop.key) for prop in self.related_keys)
        try:
            hash(key)
        except TypeError as error:
            keys = [prop.key for prop in self.related_keys]
            raise self.make_error("invalid", value=value, keys=keys) from error
        return key

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

"""Mixin that adds model instance loading behavior.

.. warning::

    This module is treated as private API.
    Users should not need to use this module directly.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import marshmallow as ma
import sqlalchemy as sa
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm.exc import ObjectDeletedError

from .fields import _MISSING, Related, RelatedList, get_primary_keys

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

_LoadDataV3 = Mapping[str, Any] | Iterable[Mapping[str, Any]]
_LoadDataV4 = Mapping[str, Any] | Sequence[Mapping[str, Any]]
_LoadDataT = TypeVar("_LoadDataT", _LoadDataV3, _LoadDataV4)
_ModelType = TypeVar("_ModelType", bound=DeclarativeMeta)


def _cast_data(data):
    if int(importlib.metadata.version("marshmallow")[0]) >= 4:
        return cast("_LoadDataV4", data)
    return cast("_LoadDataV3", data)


def _related_key_tuple(field: Related, raw: Any) -> tuple | None:
    """Normalize a raw serialized value to a related-key tuple.

    Return `None` for values that cannot be looked up in bulk; these are
    left to the regular per-item deserialization path.
    """
    props = field.related_keys
    if isinstance(raw, Mapping):
        values = tuple(raw.get(prop.key) for prop in props)
    elif len(props) == 1:
        values = (raw,)
    else:
        return None
    if any(value is None for value in values):
        return None
    try:
        hash(values)
    except TypeError:
        return None
    return values


class LoadInstanceMixin:
    class Opts:
        model: type[DeclarativeMeta] | None
        sqla_session: Session | None
        load_instance: bool
        transient: bool
        bulk_related: bool
        bulk_related_batch_size: int

        def __init__(self, meta, *args, **kwargs):
            super().__init__(meta, *args, **kwargs)
            self.model = getattr(meta, "model", None)
            self.sqla_session = getattr(meta, "sqla_session", None)
            self.load_instance = getattr(meta, "load_instance", False)
            self.transient = getattr(meta, "transient", False)
            self.bulk_related = getattr(meta, "bulk_related", False)
            self.bulk_related_batch_size = getattr(
                meta, "bulk_related_batch_size", 500
            )

    class Schema(Generic[_ModelType]):
        opts: LoadInstanceMixin.Opts
        instance: _ModelType | None
        _session: Session | None
        _transient: bool | None
        _load_instance: bool
        _related_cache: dict | None

        @property
        def session(self) -> Session | None:
            """The SQLAlchemy session used to load models."""
            return self._session or self.opts.sqla_session

        @session.setter
        def session(self, session: Session) -> None:
            self._session = session

        @property
        def transient(self) -> bool:
            """Whether model instances are loaded in a transient state."""
            if self._transient is not None:
                return self._transient
            return self.opts.transient

        @transient.setter
        def transient(self, transient: bool) -> None:
            self._transient = transient

        def __init__(self, *args, **kwargs):
            self._session = kwargs.pop("session", None)
            self.instance = kwargs.pop("instance", None)
            self._transient = kwargs.pop("transient", None)
            self._load_instance = kwargs.pop("load_instance", self.opts.load_instance)
            self._related_cache = None
            super().__init__(*args, **kwargs)

        def get_instance(self, data) -> _ModelType | None:
            """Retrieve an existing record by primary key(s). If the schema instance
            is transient, return `None`.

            :param data: Serialized data to inform lookup.
            """
            if self.transient:
                return None
            model = cast("type[_ModelType]", self.opts.model)
            props = get_primary_keys(model)
            filters = {prop.key: data.get(prop.key) for prop in props}
            if None not in filters.values():
                try:
                    return cast("Session", self.session).get(model, filters)
                except ObjectDeletedError:
                    return None
            return None

        @ma.post_load
        def make_instance(self, data, **kwargs) -> _ModelType:
            """Deserialize data to an instance of the model if self.load_instance is True.

            Update an existing row if specified in `self.instance` or loaded by primary
            key(s) in the data; else create a new row.

            :param data: Data to deserialize.
            """
            if not self._load_instance:
                return data
            instance = self.instance or self.get_instance(data)
            if instance is not None:
                for key, value in data.items():
                    setattr(instance, key, value)
                return instance
            kwargs, association_attrs = self._split_model_kwargs_association(data)
            ModelClass = cast("DeclarativeMeta", self.opts.model)
            instance = ModelClass(**kwargs)
            for attr, value in association_attrs.items():
                setattr(instance, attr, value)
            return instance

        def load(
            self,
            data: _LoadDataT,
            *,
            session: Session | None = None,
            instance: _ModelType | None = None,
            transient: bool = False,
            bulk_related: bool | None = None,
            **kwargs,
        ) -> Any:
            """Deserialize data. If ``load_instance`` is set to `True`
            in the schema meta options, load the data as model instance(s).

            :param data: The data to deserialize.
            :param session: SQLAlchemy `session <sqlalchemy.orm.Session>`.
            :param instance: Existing model instance to modify.
            :param transient: If `True`, load transient model instance(s).
            :param bulk_related: If `True`, eagerly resolve `Related` fields
                for ``many=True`` loads with batched queries instead of
                per-item lookups. Defaults to the ``bulk_related`` Meta option.
            :param kwargs: Same keyword arguments as `marshmallow.Schema.load`.
            """
            self._session = session or self._session
            self._transient = transient or self._transient
            if self._load_instance and not (self.transient or self.session):
                raise ValueError("Deserialization requires a session")
            self.instance = instance or self.instance
            try:
                if bulk_related is None:
                    bulk_related = self.opts.bulk_related
                if bulk_related:
                    self._related_cache = self._resolve_related(data, kwargs)
                return cast("ma.Schema", super()).load(_cast_data(data), **kwargs)
            finally:
                self.instance = None
                self._related_cache = None

        def _resolve_related(self, data, load_kwargs) -> dict | None:
            """Eagerly resolve `Related` fields for a ``many=True`` load.

            Collect valid related keys from the input, fetch the matching
            rows with one chunked query per related model, and return a cache
            mapping ``(group, key)`` to the resolved instance (or ``_MISSING``
            for keys without a matching row). Return `None` if bulk
            resolution does not apply. If any query fails, no partial cache
            is installed on the schema.
            """
            many = load_kwargs.get("many")
            if many is None:
                many = self.many
            if (
                not many
                or self.transient
                or self.session is None
                or not isinstance(data, Sequence)
                or isinstance(data, (str, bytes))
            ):
                return None
            related_fields = []
            for name, field in self.load_fields.items():
                if isinstance(field, Related):
                    related_fields.append((name, field, field, False))
                elif isinstance(field, RelatedList) and isinstance(
                    field.inner, Related
                ):
                    related_fields.append((name, field, field.inner, True))
            if not related_fields:
                return None
            groups: dict[tuple, dict] = {}
            for item in data:
                if not isinstance(item, Mapping):
                    continue
                for name, field, related, is_list in related_fields:
                    raw = item.get(field.data_key or name)
                    if is_list:
                        raws = raw if isinstance(raw, list) else []
                    else:
                        raws = [raw]
                    for value in raws:
                        key = _related_key_tuple(related, value)
                        if key is None:
                            continue
                        group_key = (
                            related.related_model,
                            tuple(prop.key for prop in related.related_keys),
                        )
                        group = groups.setdefault(
                            group_key, {"field": related, "keys": set()}
                        )
                        group["keys"].add(key)
            cache: dict = {}
            for group_key, group in groups.items():
                self._resolve_related_group(cache, group_key, group)
            return cache

        def _resolve_related_group(self, cache, group_key, group) -> None:
            """Resolve one group of related keys into ``cache``."""
            session = cast("Session", self.session)
            field = group["field"]
            keys = group["keys"]
            model, _ = group_key
            props = field.related_keys
            # Reuse instances already present in the session's identity map
            # when resolving by primary key, avoiding redundant SELECTs.
            if not field.columns:
                mapper = sa.inspect(model)
                for key in list(keys):
                    identity_key = mapper.identity_key_from_primary_key(list(key))
                    instance = session.identity_map.get(identity_key)
                    if instance is not None:
                        cache[(group_key, key)] = instance
                        keys.discard(key)
            if keys:
                columns = [getattr(model, prop.key) for prop in props]
                # Chunk the keys so each query stays within the batch size,
                # limiting the number of bound parameters per statement.
                chunk_size = max(1, self.opts.bulk_related_batch_size // len(columns))
                key_list = list(keys)
                for start in range(0, len(key_list), chunk_size):
                    chunk = key_list[start : start + chunk_size]
                    if len(columns) == 1:
                        condition = columns[0].in_([key[0] for key in chunk])
                    else:
                        condition = sa.tuple_(*columns).in_(chunk)
                    statement = sa.select(model).where(condition)
                    for instance in session.execute(statement).scalars():
                        found_key = tuple(getattr(instance, prop.key) for prop in props)
                        cache[(group_key, found_key)] = instance
                        keys.discard(found_key)
            # Keys without a matching row are handled per the field's
            # configuration at deserialization time.
            for key in keys:
                cache[(group_key, key)] = _MISSING

        def validate(
            self,
            data: _LoadDataT,
            *,
            session: Session | None = None,
            **kwargs,
        ) -> dict[str, list[str]]:
            """Same as `marshmallow.Schema.validate` but allows passing a ``session``."""
            self._session = session or self._session
            if not (self.transient or self.session):
                raise ValueError("Validation requires a session")
            return cast("ma.Schema", super()).validate(_cast_data(data), **kwargs)

        def _split_model_kwargs_association(self, data):
            """Split serialized attrs to ensure association proxies are passed separately.

            This is necessary for Python < 3.6.0, as the order in which kwargs are passed
            is non-deterministic, and associations must be parsed by sqlalchemy after their
            intermediate relationship, unless their `creator` has been set.

            Ignore invalid keys at this point - behaviour for unknowns should be
            handled elsewhere.

            :param data: serialized dictionary of attrs to split on association_proxy.
            """
            association_attrs = {
                key: value
                for key, value in data.items()
                # association proxy
                if hasattr(getattr(self.opts.model, key, None), "remote_attr")
            }
            kwargs = {
                key: value
                for key, value in data.items()
                if (hasattr(self.opts.model, key) and key not in association_attrs)
            }
            return kwargs, association_attrs

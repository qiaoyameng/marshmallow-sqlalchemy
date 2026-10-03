"""Mixin that adds model instance loading behavior.

.. warning::

    This module is treated as private API.
    Users should not need to use this module directly.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import marshmallow as ma
import sqlalchemy as sa
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm.exc import ObjectDeletedError

from .fields import get_primary_keys

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

_LoadDataV3 = Mapping[str, Any] | Iterable[Mapping[str, Any]]
_LoadDataV4 = Mapping[str, Any] | Sequence[Mapping[str, Any]]
_LoadDataT = TypeVar("_LoadDataT", _LoadDataV3, _LoadDataV4)
_ModelType = TypeVar("_ModelType", bound=DeclarativeMeta)


@dataclass
class _LoadPlan(Generic[_ModelType]):
    model: type[_ModelType]
    attributes: Mapping[str, Any]
    association_attrs: Mapping[str, Any]
    instance: _ModelType | None = None

    def apply(self) -> _ModelType:
        if self.instance is not None:
            instance = self.instance
            for key, value in self.attributes.items():
                setattr(instance, key, _apply_planned_value(value))
            for attr, value in self.association_attrs.items():
                setattr(instance, attr, _apply_planned_value(value))
            return instance

        instance = self.model(
            **{
                key: _apply_planned_value(value)
                for key, value in self.attributes.items()
            }
        )
        for attr, value in self.association_attrs.items():
            setattr(instance, attr, _apply_planned_value(value))
        return instance


@dataclass
class _DeferredLoad:
    schema: Any
    data: Any
    many: bool
    original_data: Any
    partial: bool | Iterable[str] | set[str] | None
    unknown: str | None

    def resolve(self) -> Any:
        data = _resolve_deferred_loads(self.data)
        previous_phase = getattr(self.schema, "_load_phase", None)
        self.schema._load_phase = "plan"
        try:
            return self.schema._invoke_load_processors(
                ma.schema.POST_LOAD,
                data,
                many=self.many,
                original_data=self.original_data,
                partial=self.partial,
                unknown=self.unknown,
            )
        finally:
            self.schema._load_phase = previous_phase


class _DeferredMappingLoad(dict):
    _deferred_load: _DeferredLoad | None


class _DeferredSequenceLoad(list):
    _deferred_load: _DeferredLoad | None


def _make_deferred_load(
    schema: ma.Schema,
    data: Any,
    *,
    many: bool,
    original_data: Any,
    partial: bool | Iterable[str] | set[str] | None,
    unknown: str | None,
) -> Any:
    deferred = _DeferredLoad(
        schema=schema,
        data=data,
        many=many,
        original_data=original_data,
        partial=partial,
        unknown=unknown,
    )
    if many:
        marker: Any = _DeferredSequenceLoad(data)
    else:
        marker = _DeferredMappingLoad(data)
    marker._deferred_load = deferred
    return marker


def _resolve_deferred_loads(value: Any) -> Any:
    deferred = getattr(value, "_deferred_load", None)
    if deferred is not None:
        return deferred.resolve()
    if isinstance(value, list):
        return [_resolve_deferred_loads(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_deferred_loads(item) for item in value)
    if isinstance(value, dict):
        return {key: _resolve_deferred_loads(item) for key, item in value.items()}
    return value


def _apply_planned_value(value: Any) -> Any:
    if isinstance(value, _LoadPlan):
        return value.apply()
    if isinstance(value, list):
        return [_apply_planned_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_apply_planned_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _apply_planned_value(item) for key, item in value.items()}
    return value


def _cast_data(data):
    if int(importlib.metadata.version("marshmallow")[0]) >= 4:
        return cast("_LoadDataV4", data)
    return cast("_LoadDataV3", data)


class LoadInstanceMixin:
    class Opts:
        model: type[DeclarativeMeta] | None
        sqla_session: Session | None
        load_instance: bool
        transient: bool

        def __init__(self, meta, *args, **kwargs):
            super().__init__(meta, *args, **kwargs)
            self.model = getattr(meta, "model", None)
            self.sqla_session = getattr(meta, "sqla_session", None)
            self.load_instance = getattr(meta, "load_instance", False)
            self.transient = getattr(meta, "transient", False)

    class Schema(Generic[_ModelType]):
        opts: LoadInstanceMixin.Opts
        instance: _ModelType | None
        _session: Session | None
        _transient: bool | None
        _load_instance: bool
        _load_phase: str | None
        _two_phase: bool

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
            self._load_phase = None
            self._two_phase = False
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
                identity = (
                    next(iter(filters.values()))
                    if len(filters) == 1
                    else tuple(filters.values())
                )
                try:
                    return cast("Session", self.session).get(model, identity)
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
            ModelClass = cast("DeclarativeMeta", self.opts.model)
            instance = self.instance or self.get_instance(data)
            attributes, association_attrs = self._split_model_kwargs_association(data)
            if self._load_phase != "plan":
                if instance is not None:
                    for key, value in attributes.items():
                        setattr(instance, key, _apply_planned_value(value))
                    for attr, value in association_attrs.items():
                        setattr(instance, attr, _apply_planned_value(value))
                    return instance
                instance = ModelClass(**attributes)
                for attr, value in association_attrs.items():
                    setattr(instance, attr, _apply_planned_value(value))
                return instance
            return _LoadPlan(
                model=cast("type[_ModelType]", ModelClass),
                attributes=attributes,
                association_attrs=association_attrs,
                instance=instance,
            )

        def _do_load(
            self,
            data,
            *,
            many: bool | None = None,
            partial=None,
            unknown=None,
            postprocess: bool = True,
        ):
            many = self.many if many is None else bool(many)
            if self._two_phase and postprocess and self._load_phase == "validate":
                result = super()._do_load(
                    _cast_data(data),
                    many=many,
                    partial=partial,
                    unknown=unknown,
                    postprocess=False,
                )
                self._validate_unique_targets(result)
                return _make_deferred_load(
                    self,
                    result,
                    many=many,
                    original_data=data,
                    partial=self.partial if partial is None else partial,
                    unknown=self.unknown if unknown is None else unknown,
                )
            return super()._do_load(
                _cast_data(data),
                many=many,
                partial=partial,
                unknown=unknown,
                postprocess=postprocess,
            )

        def load(
            self,
            data: _LoadDataT,
            *,
            session: Session | None = None,
            instance: _ModelType | None = None,
            transient: bool = False,
            two_phase: bool = False,
            **kwargs,
        ) -> Any:
            """Deserialize data. If ``load_instance`` is set to `True`
            in the schema meta options, load the data as model instance(s).

            :param data: The data to deserialize.
            :param session: SQLAlchemy `session <sqlalchemy.orm.Session>`.
            :param instance: Existing model instance to modify.
            :param transient: If `True`, load transient model instance(s).
            :param two_phase: If `True`, validate every item before applying updates.
            :param kwargs: Same keyword arguments as `marshmallow.Schema.load`.
            """
            self._session = session or self._session
            self._transient = transient or self._transient
            many = bool(kwargs.get("many", self.many))
            if self._load_instance and not (self.transient or self.session):
                raise ValueError("Deserialization requires a session")
            if two_phase and not many:
                raise ValueError("Two-phase loading requires many=True")
            if two_phase and instance is not None:
                raise ValueError(
                    "Two-phase loading cannot be combined with a single instance"
                )
            previous_instance = self.instance
            if instance is not None:
                self.instance = instance
            self._two_phase = two_phase
            try:
                if not two_phase:
                    return cast("ma.Schema", super()).load(_cast_data(data), **kwargs)

                if self.transient:
                    self._load_phase = "validate"
                    plans = cast("ma.Schema", super()).load(
                        _cast_data(data), **kwargs
                    )
                    self._load_phase = None
                    self._validate_unique_targets(plans)
                    plans = _resolve_deferred_loads(plans)
                    return _apply_planned_value(plans)

                session = cast("Session", self.session)
                with _flush_control(session, block=True):
                    self._load_phase = "validate"
                    plans = cast("ma.Schema", super()).load(
                        _cast_data(data), **kwargs
                    )
                    self._load_phase = None
                    self._validate_unique_targets(plans)
                with _flush_control(session, block=False):
                    plans = _resolve_deferred_loads(plans)
                    return _apply_planned_value(plans)
            finally:
                self._load_phase = None
                self._two_phase = False
                if instance is not None:
                    self.instance = previous_instance

        def _validate_unique_targets(self, data) -> None:
            if not self._load_instance or (
                not isinstance(data, Sequence)
                or isinstance(data, (str, bytes, Mapping))
            ):
                return

            model = cast("type[_ModelType]", self.opts.model)
            primary_keys = [prop.key for prop in get_primary_keys(model)]
            targets = {}
            errors: dict[int, list[str]] = {}
            for index, item in enumerate(data):
                if not isinstance(item, Mapping):
                    continue
                key = tuple(item.get(primary_key) for primary_key in primary_keys)
                if None in key:
                    continue
                if key in targets:
                    target = ",".join(str(value) for value in key)
                    errors.setdefault(index, []).append(
                        f"Duplicate target for {model.__name__}: {target}"
                    )
                else:
                    targets[key] = index

            if errors:
                error = ma.ValidationError(errors)
                self.handle_error(error, data, many=True)
                raise error

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


@contextlib.contextmanager
def _flush_control(session: Session, *, block: bool):
    previous_autoflush = session.autoflush
    session.autoflush = False

    def block_flush(*args, **kwargs):
        raise RuntimeError("Cannot flush during two-phase validation.")

    if block:
        sa.event.listen(session, "before_flush", block_flush)
    try:
        yield
    finally:
        if block:
            sa.event.remove(session, "before_flush", block_flush)
        session.autoflush = previous_autoflush

"""Mixin that adds model instance loading behavior.

.. warning::

    This module is treated as private API.
    Users should not need to use this module directly.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import marshmallow as ma
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm.exc import ObjectDeletedError

from .fields import get_primary_keys
from .update_plan import (
    VALID_POLICIES,
    UpdatePlan,
    _PlanContext,
    active_plan_context,
    in_nested_load,
    make_pending_instance,
    run_planning,
)

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
                session = cast("Session | None", self.session)
                with self._no_autoflush(session):
                    try:
                        return cast("Session", session).get(model, filters)
                    except ObjectDeletedError:
                        return None
            return None

        @staticmethod
        def _no_autoflush(session):
            """Suppress autoflush during planning so validation never flushes.

            Outside of a two-phase load autoflush behavior is left untouched.
            """
            if session is not None and active_plan_context() is not None:
                return session.no_autoflush
            return contextlib.nullcontext()

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
            plan_context = active_plan_context()
            if plan_context is not None:
                # Planning phase: record a validated change instead of
                # touching any mapped instance.
                return make_pending_instance(
                    cast("DeclarativeMeta", self.opts.model),
                    data,
                    instance,
                    is_root=not in_nested_load(),
                )
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
            two_phase: bool = False,
            on_duplicate: str = "raise",
            **kwargs,
        ) -> Any:
            """Deserialize data. If ``load_instance`` is set to `True`
            in the schema meta options, load the data as model instance(s).

            :param data: The data to deserialize.
            :param session: SQLAlchemy `session <sqlalchemy.orm.Session>`.
            :param instance: Existing model instance to modify.
            :param transient: If `True`, load transient model instance(s).
            :param two_phase: If `True`, validate every input record first and
                only apply the resulting, validated update plan once all
                records pass validation. Existing instances are not modified
                and the session is not flushed when validation fails.
            :param on_duplicate: Conflict strategy used by ``two_phase``
                loads when several input records resolve to the same existing
                instance. One of ``"raise"`` (the default), ``"first"``,
                ``"last"`` or ``"merge"``.
            :param kwargs: Same keyword arguments as `marshmallow.Schema.load`.
            """
            self._session = session or self._session
            self._transient = transient or self._transient
            if self._load_instance and not (self.transient or self.session):
                raise ValueError("Deserialization requires a session")
            self.instance = instance or self.instance
            if two_phase:
                if on_duplicate not in VALID_POLICIES:
                    raise ValueError(
                        "on_duplicate must be one of "
                        f"{', '.join(repr(p) for p in VALID_POLICIES)}"
                    )
                try:
                    plan = self.make_update_plan(
                        data, on_duplicate=on_duplicate, **kwargs
                    )
                    return plan.apply()
                finally:
                    self.instance = None
            plan_context = active_plan_context()
            nested_cm = (
                plan_context.nested_load()
                if plan_context is not None
                else contextlib.nullcontext()
            )
            with nested_cm:
                try:
                    return cast("ma.Schema", super()).load(
                        _cast_data(data), **kwargs
                    )
                finally:
                    self.instance = None

        def make_update_plan(
            self,
            data: _LoadDataT,
            *,
            session: Session | None = None,
            instance: _ModelType | None = None,
            transient: bool = False,
            on_duplicate: str = "raise",
            **kwargs,
        ) -> UpdatePlan:
            """Validate ``data`` and return an :class:`UpdatePlan` without
            modifying any ORM instance or flushing the session.

            All marshmallow hooks (field validators, schema validators and
            ``post_load`` processors) run exactly once during planning; the
            plan's :meth:`~UpdatePlan.apply` method performs the actual
            instance updates without re-running those hooks.

            :param data: The data to deserialize.
            :param session: SQLAlchemy `session <sqlalchemy.orm.Session>`.
            :param instance: Existing model instance a single record updates.
            :param transient: If `True`, plan transient model instance(s).
            :param on_duplicate: Strategy when several records resolve to the
                same existing instance: ``"raise"`` (default), ``"first"``,
                ``"last"`` or ``"merge"``.
            :param kwargs: Same keyword arguments as `marshmallow.Schema.load`.
            """
            if on_duplicate not in VALID_POLICIES:
                raise ValueError(
                    "on_duplicate must be one of "
                    f"{', '.join(repr(p) for p in VALID_POLICIES)}"
                )
            if not self._load_instance:
                raise ValueError(
                    "Two-phase loading requires load_instance to be enabled"
                )
            self._session = session or self._session
            self._transient = transient or self._transient
            if self._load_instance and not (self.transient or self.session):
                raise ValueError("Deserialization requires a session")
            self.instance = instance or self.instance
            many = bool(kwargs.get("many", self.many))
            context = _PlanContext(on_duplicate=on_duplicate)

            def run():
                try:
                    ma.Schema.load(self, _cast_data(data), **kwargs)
                finally:
                    self.instance = None
                return context.build(many=many)

            return run_planning(run, context)

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

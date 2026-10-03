"""Two-phase loading support: build a validated update plan before applying
it to ORM instances.

During the *planning* phase data is fully deserialized and validated, but no
mapped instance is modified and nothing is added to or flushed from the
session. The resulting :class:`UpdatePlan` can then be applied in a single,
all-or-nothing step.

``UpdatePlan`` is part of the public interface; the remaining helpers are
private API.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Mapping, TypeVar

from sqlalchemy import inspect as sa_inspect

from .exceptions import DuplicateTargetError

T = TypeVar("T")

VALID_POLICIES = ("raise", "first", "last", "merge")

_current_contexts: ContextVar[tuple["_PlanContext", ...]] = ContextVar(
    "_sqla_two_phase_contexts", default=()
)


def active_plan_context() -> "_PlanContext | None":
    """Return the innermost currently active planning context, if any."""
    contexts = _current_contexts.get()
    return contexts[-1] if contexts else None


@dataclass
class PendingInstance:
    """A validated, not-yet-applied change to a single model instance."""

    model: type
    data: dict[str, Any]
    identity: tuple[Any, ...] | None
    existing: Any | None
    ordinal: int = 0
    is_root: bool = True

    @property
    def key(self) -> tuple[type, tuple[Any, ...]] | None:
        if self.identity is None:
            return None
        return (self.model, self.identity)


@dataclass
class _Conflict:
    pending: PendingInstance
    winner: PendingInstance
    policy: str


@dataclass
class _PlanContext:
    """Accumulates pending instances for a single planning run.

    The same context is reused by nested schemas (propagated through a
    context variable), so relationship changes are materialized bottom-up
    during application.
    """

    on_duplicate: str

    pending: list[PendingInstance] = field(default_factory=list)
    registry: dict[tuple[type, tuple[Any, ...]], PendingInstance] = field(
        default_factory=dict
    )
    conflicts: list[_Conflict] = field(default_factory=list)
    nested_depth: int = 0

    @contextlib.contextmanager
    def nested_load(self):
        self.nested_depth += 1
        try:
            yield
        finally:
            self.nested_depth -= 1

    def register(self, pending: PendingInstance, *, is_root: bool) -> PendingInstance:
        """Register a pending instance, resolving duplicate targets.

        Duplicate policies only apply to top-level (root) records. Nested
        records stay independent: several parents may reference the same
        related object, and each accepted nested change is applied in input
        order to the identity-mapped instance. Records discarded by the root
        conflict policy therefore never leak their nested values.
        """
        pending.ordinal = len(self.pending)
        pending.is_root = is_root
        self.pending.append(pending)
        key = pending.key
        if key is None:
            return pending
        winner = self.registry.get(key)
        if winner is None:
            self.registry[key] = pending
            return pending
        if not is_root:
            return pending
        if self.on_duplicate == "raise":
            self.conflicts.append(_Conflict(pending, winner, "raise"))
            return pending
        if self.on_duplicate == "first":
            self.conflicts.append(_Conflict(pending, winner, "first"))
            return winner
        if self.on_duplicate == "last":
            self.conflicts.append(_Conflict(pending, winner, "last"))
            self.registry[key] = pending
            return pending
        # merge: keep the earlier pending object and fold in the new values
        self.conflicts.append(_Conflict(pending, winner, "merge"))
        winner.data.update(pending.data)
        return winner

    def build(self, *, many: bool) -> "UpdatePlan":
        for conflict in self.conflicts:
            if conflict.policy == "raise":
                pending = conflict.pending
                model_name = getattr(pending.model, "__name__", str(pending.model))
                raise DuplicateTargetError(
                    f"Duplicate target for {model_name} with primary key "
                    f"{pending.identity!r}; pass on_duplicate to choose a "
                    "conflict strategy.",
                    model=pending.model,
                    identity=pending.identity,
                )
        nested_ids = self._collect_nested_ids()
        roots = [
            pending
            for pending in self.pending
            if (
                pending.is_root
                and id(pending) not in nested_ids
                and self._is_canonical(pending)
            )
        ]
        return UpdatePlan(roots=roots, context=self, many=many)

    def _is_canonical(self, pending: PendingInstance) -> bool:
        key = pending.key
        return key is None or self.registry.get(key) is pending

    def _collect_nested_ids(self) -> set[int]:
        nested_ids: set[int] = set()
        for pending in self.pending:
            self._mark(pending.data, nested_ids)
        return nested_ids

    @staticmethod
    def _mark(value: Any, nested_ids: set[int]) -> None:
        if isinstance(value, PendingInstance):
            nested_ids.add(id(value))
            _PlanContext._mark(value.data, nested_ids)
        elif isinstance(value, Mapping):
            for subvalue in value.values():
                _PlanContext._mark(subvalue, nested_ids)
        elif isinstance(value, (list, tuple, set)):
            for subvalue in value:
                _PlanContext._mark(subvalue, nested_ids)


@dataclass
class UpdatePlan:
    """A validated set of pending model changes.

    Nothing is modified and no flush is triggered until :meth:`apply` is
    called. Existing targets are the same objects held in the session's
    identity map; new targets are instantiated at application time.
    """

    roots: list[PendingInstance]
    context: _PlanContext
    many: bool = False
    applied: bool = False
    _materialized: dict[int, Any] = field(default_factory=dict)

    def apply(self) -> Any:
        """Apply the plan and return the resulting model instances.

        Returns a single instance for a single-record plan, otherwise a list
        in input order. Idempotent: a second call returns the same instances
        without re-running schema hooks or re-applying changes.
        """
        results = [self._materialize(root) for root in self.roots]
        self.applied = True
        if self.many:
            return results
        return results[0]

    def _materialize(self, pending: PendingInstance) -> Any:
        instance = self._materialized.get(id(pending))
        if instance is not None:
            return instance
        data = self._resolve(pending.data)
        instance = pending.existing
        if instance is None:
            kwargs, association_attrs = _split_model_kwargs_association(
                pending.model, data
            )
            instance = pending.model(**kwargs)
            for attr, value in association_attrs.items():
                setattr(instance, attr, value)
        else:
            for key, value in data.items():
                setattr(instance, key, value)
        self._materialized[id(pending)] = instance
        return instance

    def _resolve(self, value: Any) -> Any:
        if isinstance(value, PendingInstance):
            return self._materialize(value)
        if isinstance(value, Mapping):
            return {key: self._resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._resolve(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._resolve(item) for item in value)
        if isinstance(value, set):
            return {self._resolve(item) for item in value}
        return value


def _split_model_kwargs_association(model: type, data: Mapping[str, Any]):
    """Split serialized attrs so association proxies are passed separately."""
    association_attrs = {
        key: value
        for key, value in data.items()
        if hasattr(getattr(model, key, None), "remote_attr")
    }
    kwargs = {
        key: value
        for key, value in data.items()
        if hasattr(model, key) and key not in association_attrs
    }
    return kwargs, association_attrs


def make_pending_instance(
    model: type, data: Mapping[str, Any], existing: Any | None, *, is_root: bool
) -> PendingInstance:
    """Create and register a pending instance in the active planning context.

    When the target was already planned and the duplicate policy aliases
    records, the winning pending object is returned instead.
    """
    context = active_plan_context()
    if context is None:
        raise RuntimeError("No active two-phase load context")
    if existing is not None:
        identity = state_identity(existing)
    else:
        identity = identity_for(model, data)
    pending = PendingInstance(
        model=model, data=dict(data), identity=identity, existing=existing
    )
    return context.register(pending, is_root=is_root)


def identity_for(model: type, data: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Extract the primary key tuple for ``model`` from deserialized data."""
    mapper = sa_inspect(model)
    values = tuple(
        data.get(prop.key)
        for prop in (
            mapper.get_property_by_column(column) for column in mapper.primary_key
        )
    )
    if any(value is None for value in values):
        return None
    return values


def state_identity(instance: Any) -> tuple[Any, ...] | None:
    """Return the persistent identity tuple of a mapped instance, if any."""
    identity = sa_inspect(instance).identity
    if identity is None or any(value is None for value in identity):
        return None
    return tuple(identity)


def run_planning(action: Callable[[], T], context: _PlanContext) -> T:
    """Run ``action`` with ``context`` installed on the context variable."""
    token = _current_contexts.set(_current_contexts.get() + (context,))
    try:
        return action()
    finally:
        _current_contexts.reset(token)


def in_nested_load() -> bool:
    """Whether planning is currently inside a nested schema's load."""
    contexts = _current_contexts.get()
    if not contexts:
        return False
    return contexts[-1].nested_depth > 0

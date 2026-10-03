"""Support for serializing only data already loaded onto ORM instances.

When ``loaded_only`` is enabled on a `SQLAlchemySchema`, dumping inspects the
SQLAlchemy ``InstanceState`` of each object instead of triggering lazy loads.
Attributes that are not present (deferred columns, expired attributes or
unloaded relationships) are either skipped or reported, depending on the
configured ``unloaded`` strategy.

None of the checks in this module access values through the ORM descriptors, so
they never emit SQL -- even for detached instances.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Literal

import sqlalchemy as sa

from .exceptions import UnloadedAttributeError

#: Raise `UnloadedAttributeError` when an attribute has not been loaded.
RAISE: Literal["raise"] = "raise"
#: Omit unloaded attributes from the serialized output.
OMIT: Literal["omit"] = "omit"

UnloadedStrategy = Literal["raise", "omit"]
_STRATEGIES = (RAISE, OMIT)

# Sentinel returned in place of a value when an attribute must be omitted. It
# is distinct from marshmallow's ``missing`` so that field ``dump_default``
# values cannot be mistaken for real, database-loaded data.
class _Omitted:
    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return "<unloaded-attribute-omitted>"


OMITTED = _Omitted()

# Strategy of the outermost loaded-only dump. Nested schemas inherit it.
_current_strategy: ContextVar[UnloadedStrategy | None] = ContextVar(
    "sqla_loaded_only_strategy", default=None
)
# Field path segments accumulated during the active dump.
_current_path: ContextVar[tuple[str | int, ...]] = ContextVar(
    "sqla_loaded_only_path", default=()
)


def validate_strategy(value: Any) -> UnloadedStrategy:
    if value not in _STRATEGIES:
        raise ValueError(
            f"`unloaded` must be one of {list(_STRATEGIES)!r}, got {value!r}."
        )
    return value  # type: ignore[return-value]


def format_path(segments: tuple[str | int, ...]) -> str:
    path = ""
    for segment in segments:
        part = f"[{segment}]" if isinstance(segment, int) else str(segment)
        if part.startswith("["):
            path += part
        elif path:
            path += f".{part}"
        else:
            path = part
    return path


@contextmanager
def dump_context(strategy: UnloadedStrategy) -> Iterator[None]:
    """Activate the loaded-only mode for a (possibly nested) dump.

    The strategy of the outermost context wins, so nested schemas -- including
    plain ``marshmallow.Nested`` wrappers -- transparently inherit it. The
    field path is never reset: segments pushed by surrounding nested fields
    (e.g. a plain marshmallow parent) are preserved.
    """
    strategy_token = None
    if _current_strategy.get() is None:
        strategy_token = _current_strategy.set(strategy)
    try:
        yield
    finally:
        if strategy_token is not None:
            _current_strategy.reset(strategy_token)


def current_strategy() -> UnloadedStrategy | None:
    return _current_strategy.get()


def current_path() -> tuple[str | int, ...]:
    return _current_path.get()


@contextmanager
def path_segment(segment: str | int) -> Iterator[None]:
    token = _current_path.set((*_current_path.get(), segment))
    try:
        yield
    finally:
        _current_path.reset(token)


def _orm_state(obj: Any):
    try:
        state = sa.inspect(obj)
    except sa.exc.NoInspectionAvailable:
        return None
    if not isinstance(state, sa.orm.state.InstanceState):
        return None
    return state


def _resolve_mapped_key(state, attr: str) -> str | None:
    """Map an output attribute to the underlying instrumented key.

    Association proxies resolve to their local (usually relationship)
    attribute. Returns ``None`` when the attribute is unmapped and therefore
    cannot trigger a lazy load on its own.
    """
    mapper = state.mapper
    try:
        if attr in mapper.attrs:
            return attr
    except KeyError:
        pass
    descriptor = mapper.all_orm_descriptors.get(attr)
    # Association proxies: SQLAlchemy >=2.0 stores `target_collection`;
    # older versions expose the local InstrumentedAttribute as `local_attr`.
    target_collection = getattr(descriptor, "target_collection", None)
    if target_collection is not None:
        return target_collection
    local_attr = getattr(descriptor, "local_attr", None)
    if local_attr is not None:
        return local_attr.key
    return None


def unloaded_reason(state, mapped_key: str) -> str | None:
    """Return why ``mapped_key`` is unavailable, or ``None`` if it is loaded."""
    if mapped_key in state.expired_attributes:
        return "the attribute is expired"
    if mapped_key in state.unloaded:
        return "the attribute has not been loaded"
    return None


def make_error(obj: Any, reason: str, extra: tuple[str, ...] = ()):
    state = _orm_state(obj)
    lifecycle = (
        " (the instance is detached from its session)"
        if state is not None and state.detached
        else ""
    )
    path = format_path((*current_path(), *extra))
    return UnloadedAttributeError(
        f"Cannot serialize {path!r}: {reason}{lifecycle}. Load the attribute "
        "before dumping, or use unloaded='omit' to skip unloaded attributes."
    )


def get_loaded_value(obj: Any, attr: str, extra: str | None = None):
    """Return the loaded value for ``attr`` without triggering a lazy load.

    The active path is assumed to already identify the outer field. ``extra``
    may append a relative segment (used when a `Related` field reads several
    columns off a related object).

    - A loaded value (including a loaded ``NULL``) is returned as-is.
    - `OMITTED` is returned for a missing attribute under the ``"omit"``
      strategy.
    - `UnloadedAttributeError` is raised under the ``"raise"`` strategy.
    - Non-ORM objects and unmapped attributes return marshmallow's ``missing``
      so callers fall back to ordinary attribute access.
    """
    from marshmallow.utils import missing as marshal_missing

    state = _orm_state(obj)
    if state is None:
        return marshal_missing
    mapped_key = _resolve_mapped_key(state, attr)
    if mapped_key is None:
        return marshal_missing
    reason = unloaded_reason(state, mapped_key)
    if reason is None:
        if mapped_key == attr:
            # Read the cached value directly: this never triggers a lazy load.
            return state.attrs[mapped_key].loaded_value
        # The output attribute (typically an association proxy) maps to a
        # different mapped key. Its local collection is confirmed loaded, so
        # evaluating the proxy itself cannot emit SQL.
        return getattr(obj, attr)
    strategy = current_strategy()
    if strategy is None:
        return marshal_missing
    if strategy == OMIT:
        return OMITTED
    raise make_error(obj, reason, extra=(extra,) if extra else ())

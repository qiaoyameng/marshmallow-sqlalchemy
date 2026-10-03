"""Support for "loaded-only" dumps.

When :class:`SQLAlchemySchema <marshmallow_sqlalchemy.SQLAlchemySchema>` is
configured with ``dump_loaded_only = True``, serialization reads only the
data already present on the SQLAlchemy instance. Attributes that would trigger
a lazy SQL query on access -- deferred columns, expired attributes and
unloaded relationships -- are either omitted from the output or reported
through :class:`UnloadedAttributeError`.

.. warning::

    This module is treated as private API.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, NamedTuple

from sqlalchemy import inspect
from sqlalchemy.orm.state import InstanceState

#: Raise :class:`UnloadedAttributeError` when an unloaded attribute is met.
RAISE = "raise"

#: Silently skip fields backed by an unloaded attribute.
OMIT = "omit"

VALID_STRATEGIES = (RAISE, OMIT)


class LoadedOnlySettings(NamedTuple):
    enabled: bool
    strategy: str


#: Effective settings for the dump currently in progress. Nested schemas
#: consult this so that a strategy selected on the root schema is inherited
#: by every nested schema involved in the same dump.
_dump_context: ContextVar[LoadedOnlySettings | None] = ContextVar(
    "_loaded_only_dump_context", default=None
)


def is_mapped_instance(obj: Any) -> bool:
    """Return whether ``obj`` is a mapped SQLAlchemy instance whose ORM
    state can be inspected without triggering any SQL."""
    return isinstance(inspect(obj, raiseerr=False), InstanceState)


def is_unloaded(obj: Any, attribute: str) -> bool:
    """Return whether ``attribute`` on ``obj`` is mapped but not loaded.

    The check never touches ``getattr``: it only reads the instance's
    :class:`InstanceState`, so no lazy query can be triggered and a detached
    instance does not raise ``DetachedInstanceError``.

    A loaded value -- including a loaded ``NULL`` -- is considered loaded.
    """
    state: InstanceState = inspect(obj, raiseerr=False)
    if not isinstance(state, InstanceState):
        return False
    if attribute not in state.mapper.attrs:
        # Attribute not mapped by the ORM (e.g. an association proxy or a
        # plain Python attribute): there is nothing SQLAlchemy would load.
        return False
    return attribute in state.unloaded


def detached_hint(obj: Any) -> str | None:
    """Describe the instance's ORM state for error messages."""
    state: InstanceState = inspect(obj, raiseerr=False)
    if not isinstance(state, InstanceState):
        return None
    if state.detached:
        return "the instance is detached from its session"
    if state.expired:
        return "the instance is expired"
    return None

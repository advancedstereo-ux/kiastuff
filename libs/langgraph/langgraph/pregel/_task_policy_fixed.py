"""Fixed duplicate of ``_task_policy.py``.

Differences from the original:

* ``assert_pull_route_guards`` now also enforces ``task_node_allowlist`` in
  addition to ``route_guards``. This closes the bypass where a node returning
  ``Command(goto="<str>")`` emitted a ``branch:to:<target>`` channel write that
  triggered the target as a PULL task without an allowlist check.

Every other helper is reproduced verbatim. Do not edit the original file;
import from this module where the fix is required, e.g.:

    from langgraph.pregel._task_policy_fixed import assert_pull_route_guards
"""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig

from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_READ,
    CONFIG_KEY_ROUTE_GUARDS,
    CONFIG_KEY_ROUTE_STAMPS,
    CONFIG_KEY_TASK_NS,
    CONFIG_KEY_TASK_NODE_ALLOWLIST,
    CONFIG_KEY_TASK_TOOL_ALLOWLIST,
)
from langgraph.errors import InvalidUpdateError


def get_task_node_allowlist(config: RunnableConfig | None) -> frozenset[str] | None:
    if not config:
        return None
    raw = config.get(CONF, {}).get(CONFIG_KEY_TASK_NODE_ALLOWLIST)
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)):
        raise TypeError(
            "task_node_allowlist must be a non-string sequence of node names, "
            f"got {type(raw).__name__}"
        )
    if not isinstance(raw, Sequence):
        raise TypeError(
            "task_node_allowlist must be a sequence of node names, "
            f"got {type(raw).__name__}"
        )
    items = [str(x) for x in raw]
    if not items:
        return None
    return frozenset(items)


def get_route_guards(config: RunnableConfig | None) -> dict[str, tuple[str, ...]] | None:
    if not config:
        return None
    raw = config.get(CONF, {}).get(CONFIG_KEY_ROUTE_GUARDS)
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)):
        raise TypeError(
            "route_guards must be a mapping from node name to a sequence of channel keys, "
            f"got {type(raw).__name__}"
        )
    if not isinstance(raw, Mapping):
        raise TypeError(
            "route_guards must be a mapping from node name to a sequence of channel keys, "
            f"got {type(raw).__name__}"
        )
    out: dict[str, tuple[str, ...]] = {}
    for node, reqs in raw.items():
        if isinstance(reqs, (str, bytes)):
            raise TypeError(
                "route_guards values must be non-string sequences of channel key names"
            )
        if not isinstance(reqs, Sequence):
            raise TypeError(
                "route_guards values must be sequences of channel key names, "
                f"got {type(reqs).__name__}"
            )
        keys = tuple(str(x) for x in reqs)
        if keys:
            out[str(node)] = keys
    return out or None


def extend_writes_with_route_stamps(
    *,
    task_name: str,
    writes: MutableSequence[tuple[str, Any]],
    config: RunnableConfig | None,
) -> None:
    if not config:
        return
    raw = config.get(CONF, {}).get(CONFIG_KEY_ROUTE_STAMPS)
    if raw is None:
        return
    if isinstance(raw, (str, bytes)):
        raise TypeError(
            "route_stamps must be a mapping from node name to channel updates, "
            f"got {type(raw).__name__}"
        )
    if not isinstance(raw, Mapping):
        raise TypeError(
            "route_stamps must be a mapping from node name to channel updates, "
            f"got {type(raw).__name__}"
        )
    spec = raw.get(task_name)
    if spec is None:
        return
    if isinstance(spec, Mapping):
        for channel, value in spec.items():
            writes.append((str(channel), value))
    elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        for channel in spec:
            writes.append((str(channel), True))
    else:
        raise TypeError(
            "route_stamps values must be a mapping channel->value or a non-string "
            f"sequence of channel names, got {type(spec).__name__} for node {task_name!r}"
        )


def _assert_allowlist(config: RunnableConfig | None, target_node: str) -> None:
    allowlist = get_task_node_allowlist(config)
    if allowlist is None:
        return
    if target_node in allowlist:
        return
    task_ns = (config or {}).get(CONF, {}).get(CONFIG_KEY_TASK_NS)
    suffix = f" (task_ns={task_ns!r})" if task_ns is not None else ""
    raise InvalidUpdateError(
        "Task policy denies routing to node "
        f"{target_node!r}: not in task_node_allowlist{suffix}"
    )


def assert_route_guards_satisfied(
    config: RunnableConfig | None, target_node: str
) -> None:
    guards = get_route_guards(config)
    if not guards:
        return
    required = guards.get(target_node)
    if not required:
        return
    read_fn = (config or {}).get(CONF, {}).get(CONFIG_KEY_READ)
    if read_fn is None:
        raise InvalidUpdateError(
            "route_guards is set but no state read function is available in config; "
            "routing with guards must run inside the Pregel loop (or supply configurable read)."
        )
    snapshot = read_fn(list(required), False)
    if not isinstance(snapshot, dict):
        raise InvalidUpdateError(
            "route_guards: read(keys) must return a dict when reading multiple channels"
        )
    task_ns = (config or {}).get(CONF, {}).get(CONFIG_KEY_TASK_NS)
    suffix = f" (task_ns={task_ns!r})" if task_ns is not None else ""
    for k in required:
        if k not in snapshot or not snapshot[k]:
            raise InvalidUpdateError(
                "Route guard denies routing to node "
                f"{target_node!r}: required state key {k!r} is missing or false{suffix}"
            )


def assert_task_routing_allowed(
    config: RunnableConfig | None, target_node: str
) -> None:
    _assert_allowlist(config, target_node)
    assert_route_guards_satisfied(config, target_node)


def assert_pull_route_guards(
    config: RunnableConfig | None, target_node: str
) -> None:
    """FIXED: enforce both ``task_node_allowlist`` and ``route_guards`` on PULL.

    The original implementation only checked guards, which allowed a node to
    bypass namespace isolation by returning ``Command(goto="<foreign_node>")``
    with a string target — that path emits a ``branch:to:<target>`` channel
    write and schedules the target as a PULL task. With this fix, the
    allowlist is enforced on every dynamic routing path (PUSH, PULL, and
    top-level Command input) uniformly.
    """
    _assert_allowlist(config, target_node)
    assert_route_guards_satisfied(config, target_node)


def task_node_visible_for_discovery(
    node_name: str,
    config: RunnableConfig | None,
    *,
    list_mode: bool,
) -> bool:
    if not list_mode:
        return True
    allowlist = get_task_node_allowlist(config)
    if allowlist is None:
        return True
    return node_name in allowlist


def get_task_tool_allowlist(config: RunnableConfig | None) -> frozenset[str] | None:
    if not config:
        return None
    raw = config.get(CONF, {}).get(CONFIG_KEY_TASK_TOOL_ALLOWLIST)
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)):
        raise TypeError(
            "task_tool_allowlist must be a non-string sequence of tool names, "
            f"got {type(raw).__name__}"
        )
    if not isinstance(raw, Sequence):
        raise TypeError(
            "task_tool_allowlist must be a sequence of tool names, "
            f"got {type(raw).__name__}"
        )
    items = [str(x) for x in raw]
    if not items:
        return None
    return frozenset(items)


def parse_thread_id_scope(thread_id: str) -> tuple[str, str, str, str]:
    parts = thread_id.split(":")
    if len(parts) != 4 or any(not p for p in parts):
        raise InvalidUpdateError(
            "thread_id must use format 'tenant_id:user_id:session_id:run_id'."
        )
    return parts[0], parts[1], parts[2], parts[3]


def thread_id_scope_key(thread_id: str) -> str:
    tenant_id, user_id, session_id, _ = parse_thread_id_scope(thread_id)
    return f"{tenant_id}:{user_id}:{session_id}"

"""Task namespace allowlist (routing + discovery listing)."""

from typing_extensions import TypedDict

import pytest

from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_ENFORCE_THREAD_ID_FORMAT,
    CONFIG_KEY_ENFORCE_THREAD_ID_SCOPE,
    CONFIG_KEY_READ,
    CONFIG_KEY_ROUTE_GUARDS,
    CONFIG_KEY_ROUTE_STAMPS,
    CONFIG_KEY_TASK_NODE_ALLOWLIST,
    CONFIG_KEY_TASK_NS,
    CONFIG_KEY_TASK_TOOL_ALLOWLIST,
    THREAD_ID_SCOPE_BINDING,
)
from langgraph.errors import InvalidUpdateError
from langgraph.checkpoint.memory import InMemorySaver


def _corrupt_thread_scope_binding(
    saver: InMemorySaver, thread_id: str, wrong_scope: str
) -> None:
    """Force metadata scope mismatch for tests (simulates tampered checkpoint)."""
    for _ns, checkpoints in saver.storage[thread_id].items():
        for cp_id, (chk_ser, meta_ser, parent) in list(checkpoints.items()):
            meta = saver.serde.loads_typed(meta_ser)
            meta[THREAD_ID_SCOPE_BINDING] = wrong_scope
            checkpoints[cp_id] = (chk_ser, saver.serde.dumps_typed(meta), parent)
from langgraph.graph import END, START, StateGraph
from langgraph.pregel._io import map_command
from langgraph.pregel._task_policy import (
    assert_route_guards_satisfied,
    assert_task_routing_allowed,
    extend_writes_with_route_stamps,
    get_route_guards,
    get_task_node_allowlist,
    get_task_tool_allowlist,
    parse_thread_id_scope,
    task_node_visible_for_discovery,
)
from langgraph.types import Command, Send


def test_get_task_node_allowlist_empty_means_no_enforcement() -> None:
    assert get_task_node_allowlist({CONF: {CONFIG_KEY_TASK_NODE_ALLOWLIST: []}}) is None
    assert get_task_node_allowlist({CONF: {}}) is None


def test_assert_task_routing_allowed() -> None:
    cfg = {CONF: {CONFIG_KEY_TASK_NODE_ALLOWLIST: ["a"], CONFIG_KEY_TASK_NS: "t1"}}
    assert_task_routing_allowed(cfg, "a")
    with pytest.raises(InvalidUpdateError, match="denies routing"):
        assert_task_routing_allowed(cfg, "b")


def test_route_guards_denies_without_truthy_state_key() -> None:
    class _RG(TypedDict, total=False):
        ok: bool

    def route_a(state: _RG):
        return Command(goto="b")

    def node_b(state: _RG):
        return state

    g = (
        StateGraph(_RG)
        .add_node("a", route_a)
        .add_node("b", node_b)
        .add_edge(START, "a")
        .add_edge("b", END)
        .compile(checkpointer=InMemorySaver())
    )
    guards_cfg = {CONFIG_KEY_ROUTE_GUARDS: {"b": ["ok"]}}
    with pytest.raises(InvalidUpdateError, match="Route guard"):
        g.invoke(
            {},
            {"configurable": {"thread_id": "tg-deny", **guards_cfg}},
        )

    out = g.invoke(
        {"ok": True},
        {"configurable": {"thread_id": "tg-ok", **guards_cfg}},
    )
    assert out.get("ok") is True


def test_route_stamps_merge_on_success_so_guard_passes() -> None:
    class _RG(TypedDict, total=False):
        ok: bool

    def route_a(state: _RG):
        return Command(goto="b")

    def node_b(state: _RG):
        return state

    g = (
        StateGraph(_RG)
        .add_node("a", route_a)
        .add_node("b", node_b)
        .add_edge(START, "a")
        .add_edge("b", END)
        .compile(checkpointer=InMemorySaver())
    )
    policy = {
        CONFIG_KEY_ROUTE_GUARDS: {"b": ["ok"]},
        CONFIG_KEY_ROUTE_STAMPS: {"a": {"ok": True}},
    }
    out = g.invoke({}, {"configurable": {"thread_id": "rs-pass", **policy}})
    assert out.get("ok") is True


def test_route_stamps_sequence_sets_channels_truthy() -> None:
    class _RG(TypedDict, total=False):
        ok: bool

    def route_a(state: _RG):
        return Command(goto="b")

    def node_b(state: _RG):
        return state

    g = (
        StateGraph(_RG)
        .add_node("a", route_a)
        .add_node("b", node_b)
        .add_edge(START, "a")
        .add_edge("b", END)
        .compile(checkpointer=InMemorySaver())
    )
    policy = {
        CONFIG_KEY_ROUTE_GUARDS: {"b": ["ok"]},
        CONFIG_KEY_ROUTE_STAMPS: {"a": ("ok",)},
    }
    out = g.invoke({}, {"configurable": {"thread_id": "rs-seq", **policy}})
    assert out.get("ok") is True


def test_extend_writes_with_route_stamps_appends() -> None:
    from collections import deque

    w: deque[tuple[str, object]] = deque([("__pregel_tasks", object())])
    extend_writes_with_route_stamps(
        task_name="n",
        writes=w,
        config={CONF: {CONFIG_KEY_ROUTE_STAMPS: {"n": {"flag": 1}}}},
    )
    assert list(w)[-1] == ("flag", 1)


def test_assert_route_guards_with_read_fn() -> None:
    def read(keys, _fresh=False):
        return {k: {"ok": True, "x": 1}.get(k) for k in keys}

    cfg = {
        CONF: {
            CONFIG_KEY_ROUTE_GUARDS: {"n": ["ok"]},
            CONFIG_KEY_READ: read,
        }
    }
    assert_route_guards_satisfied(cfg, "n")
    bad = {
        CONF: {
            CONFIG_KEY_ROUTE_GUARDS: {"n": ["ok"]},
            CONFIG_KEY_READ: lambda keys, _f: {k: False for k in keys},
        }
    }
    with pytest.raises(InvalidUpdateError, match="Route guard"):
        assert_route_guards_satisfied(bad, "n")


def test_get_route_guards_none_when_unset() -> None:
    assert get_route_guards({CONF: {}}) is None


def test_map_command_respects_allowlist() -> None:
    allow = {CONF: {CONFIG_KEY_TASK_NODE_ALLOWLIST: ["n"]}}
    list(
        map_command(Command(goto=Send("n", {"k": 1})), config=allow)
    )  # does not raise
    with pytest.raises(InvalidUpdateError):
        list(map_command(Command(goto=Send("other", {})), config=allow))
    with pytest.raises(InvalidUpdateError):
        list(map_command(Command(goto="other"), config=allow))


def test_map_command_no_allowlist_unchanged() -> None:
    out = list(map_command(Command(goto=Send("any", {}))))
    assert any(c == "__pregel_tasks" for _, c, _ in out)


class _S(TypedDict):
    x: int


def test_get_subgraphs_listing_filters_by_allowlist() -> None:
    inner = (
        StateGraph(_S)
        .add_node("a", lambda s: s)
        .add_edge(START, "a")
        .add_edge("a", END)
        .compile()
    )
    parent = (
        StateGraph(_S)
        .add_node("inner", inner)
        .add_edge(START, "inner")
        .add_edge("inner", END)
        .compile()
    )
    all_sg = list(parent.get_subgraphs())
    assert len(all_sg) >= 1
    hidden = {
        CONF: {CONFIG_KEY_TASK_NODE_ALLOWLIST: ["not_a_subgraph_node"]},
    }
    assert list(parent.get_subgraphs(config=hidden)) == []


def test_get_subgraphs_navigation_ignores_allowlist() -> None:
    inner = (
        StateGraph(_S)
        .add_node("a", lambda s: s)
        .add_edge(START, "a")
        .add_edge("a", END)
        .compile()
    )
    parent = (
        StateGraph(_S)
        .add_node("inner", inner)
        .add_edge(START, "inner")
        .add_edge("inner", END)
        .compile()
    )
    cfg = {CONF: {CONFIG_KEY_TASK_NODE_ALLOWLIST: ["other"]}}
    found = list(parent.get_subgraphs(namespace="inner", recurse=False, config=cfg))
    assert len(found) == 1
    assert found[0][0] == "inner"


def test_task_node_visible_for_discovery() -> None:
    cfg = {CONF: {CONFIG_KEY_TASK_NODE_ALLOWLIST: ["a"]}}
    assert task_node_visible_for_discovery("a", cfg, list_mode=True) is True
    assert task_node_visible_for_discovery("b", cfg, list_mode=True) is False
    assert task_node_visible_for_discovery("b", cfg, list_mode=False) is True


def test_get_task_tool_allowlist() -> None:
    assert get_task_tool_allowlist({CONF: {CONFIG_KEY_TASK_TOOL_ALLOWLIST: ["x"]}}) == frozenset({"x"})
    assert get_task_tool_allowlist({CONF: {}}) is None


def test_parse_thread_id_scope() -> None:
    assert parse_thread_id_scope("t:u:s:r") == ("t", "u", "s", "r")
    with pytest.raises(InvalidUpdateError, match="thread_id must use format"):
        parse_thread_id_scope("bad")


def test_validate_rejects_cross_task_edge() -> None:
    g = StateGraph(_S)
    g.add_node("a", lambda s: s, task_ns="t1")
    g.add_node("b", lambda s: s, task_ns="t2")
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    with pytest.raises(ValueError, match="Cross-task edge"):
        g.compile()


def test_validate_allows_orchestrator_cross_task_edge() -> None:
    g = StateGraph(_S)
    g.add_node("a", lambda s: s, task_ns="t1")
    g.add_node("b", lambda s: s, task_ns="t2")
    g.add_edge(START, "a")
    g.add_edge("a", "b", allow_cross_task=True)
    g.add_edge("b", END)
    g.compile()


def test_nodes_for_task_ns() -> None:
    g = StateGraph(_S)
    g.add_node("a", lambda s: s, task_ns="t1")
    g.add_node("b", lambda s: s, task_ns="t2")
    assert g.nodes_for_task_ns("t1") == frozenset({"a"})
    assert g.nodes_for_task_ns("t2") == frozenset({"b"})


def test_auto_task_allowlist_merges_from_task_ns() -> None:
    g = StateGraph(_S)
    g.add_node("a", lambda s: s, task_ns="t1")
    g.add_node("b", lambda s: s, task_ns="t2")
    g.add_edge(START, "a")
    g.add_edge("a", END)
    compiled = g.compile(auto_task_allowlist=True)
    out = compiled._with_auto_task_allowlist({"configurable": {"task_ns": "t1"}})
    assert out["configurable"][CONFIG_KEY_TASK_NODE_ALLOWLIST] == ["a"]

    compiled_off = g.compile(auto_task_allowlist=False)
    out2 = compiled_off._with_auto_task_allowlist({"configurable": {"task_ns": "t1"}})
    assert CONFIG_KEY_TASK_NODE_ALLOWLIST not in out2.get("configurable", {})


def test_auto_task_allowlist_respects_explicit_allowlist() -> None:
    g = StateGraph(_S)
    g.add_node("a", lambda s: s, task_ns="t1")
    g.add_edge(START, "a")
    g.add_edge("a", END)
    compiled = g.compile(auto_task_allowlist=True)
    out = compiled._with_auto_task_allowlist(
        {"configurable": {"task_ns": "t1", CONFIG_KEY_TASK_NODE_ALLOWLIST: ["z"]}}
    )
    assert out["configurable"][CONFIG_KEY_TASK_NODE_ALLOWLIST] == ["z"]


def test_enforce_thread_task_ns_binds_and_rejects_mismatch() -> None:
    def node(state: _S) -> _S:
        return {"x": state["x"]}

    g = (
        StateGraph(_S)
        .add_node("a", node, task_ns="t1")
        .add_edge(START, "a")
        .add_edge("a", END)
        .compile(checkpointer=InMemorySaver())
    )

    g.invoke(
        {"x": 1},
        {"configurable": {"thread_id": "th", CONFIG_KEY_TASK_NS: "t1", "enforce_thread_task_ns": True}},
        durability="sync",
    )

    with pytest.raises(InvalidUpdateError, match="thread_id is bound"):
        g.invoke(
            {"x": 1},
            {"configurable": {"thread_id": "th", CONFIG_KEY_TASK_NS: "t2", "enforce_thread_task_ns": True}},
            durability="sync",
        )


def test_enforce_thread_task_ns_allows_same_task_ns() -> None:
    def node(state: _S) -> _S:
        return {"x": state["x"]}

    checkpointer = InMemorySaver()
    g = (
        StateGraph(_S)
        .add_node("a", node, task_ns="t1")
        .add_edge(START, "a")
        .add_edge("a", END)
        .compile(checkpointer=checkpointer)
    )

    g.invoke(
        {"x": 1},
        {"configurable": {"thread_id": "th", CONFIG_KEY_TASK_NS: "t1", "enforce_thread_task_ns": True}},
        durability="sync",
    )
    out = g.invoke(
        {"x": 2},
        {"configurable": {"thread_id": "th", CONFIG_KEY_TASK_NS: "t1", "enforce_thread_task_ns": True}},
        durability="sync",
    )
    assert out["x"] == 2


def test_enforce_thread_id_format_rejects_bad_thread_id() -> None:
    g = (
        StateGraph(_S)
        .add_node("a", lambda s: s)
        .add_edge(START, "a")
        .add_edge("a", END)
        .compile(checkpointer=InMemorySaver())
    )

    with pytest.raises(InvalidUpdateError, match="thread_id must use format"):
        g.invoke(
            {"x": 1},
            {
                "configurable": {
                    "thread_id": "bad",
                    CONFIG_KEY_ENFORCE_THREAD_ID_FORMAT: True,
                }
            },
            durability="sync",
        )


def test_enforce_thread_id_scope_rejects_mismatch() -> None:
    """Scope is stored on the checkpoint; same thread_id must match bound tenant:user:session."""
    checkpointer = InMemorySaver()
    g = (
        StateGraph(_S)
        .add_node("a", lambda s: s)
        .add_edge(START, "a")
        .add_edge("a", END)
        .compile(checkpointer=checkpointer)
    )

    thread_id = "tenant1:user1:session1:run1"
    g.invoke(
        {"x": 1},
        {
            "configurable": {
                "thread_id": thread_id,
                CONFIG_KEY_ENFORCE_THREAD_ID_SCOPE: True,
            }
        },
        durability="sync",
    )

    _corrupt_thread_scope_binding(
        checkpointer, thread_id, "tenant2:user1:session1"
    )

    with pytest.raises(InvalidUpdateError, match="thread_id scope is bound"):
        g.invoke(
            {"x": 1},
            {
                "configurable": {
                    "thread_id": thread_id,
                    CONFIG_KEY_ENFORCE_THREAD_ID_SCOPE: True,
                }
            },
            durability="sync",
        )


def test_enforce_thread_id_scope_allows_continuation_same_thread_id() -> None:
    """Same thread_id resumes the same checkpoint; scope binding stays consistent."""
    checkpointer = InMemorySaver()
    g = (
        StateGraph(_S)
        .add_node("a", lambda s: s)
        .add_edge(START, "a")
        .add_edge("a", END)
        .compile(checkpointer=checkpointer)
    )

    thread_id = "tenant1:user1:session1:run1"
    g.invoke(
        {"x": 1},
        {
            "configurable": {
                "thread_id": thread_id,
                CONFIG_KEY_ENFORCE_THREAD_ID_SCOPE: True,
            }
        },
        durability="sync",
    )
    out = g.invoke(
        {"x": 2},
        {
            "configurable": {
                "thread_id": thread_id,
                CONFIG_KEY_ENFORCE_THREAD_ID_SCOPE: True,
            }
        },
        durability="sync",
    )
    assert out["x"] == 2


def test_enforce_thread_id_format_allows_valid_thread_id() -> None:
    g = (
        StateGraph(_S)
        .add_node("a", lambda s: s)
        .add_edge(START, "a")
        .add_edge("a", END)
        .compile(checkpointer=InMemorySaver())
    )
    out = g.invoke(
        {"x": 3},
        {
            "configurable": {
                "thread_id": "tenant_a:user_b:session_c:run_d",
                CONFIG_KEY_ENFORCE_THREAD_ID_FORMAT: True,
            }
        },
        durability="sync",
    )
    assert out["x"] == 3


def test_expand_shared_nodes_blocks_delegate_to_other_task_proxy() -> None:
    def t1_agent(state: _S):
        # Delegate to the t2 proxy; with task_ns=t1, this should be denied.
        return Command(goto=Send("shared__proxy__t2", state))

    def shared_fn(state: _S):
        return {"x": state["x"] + 10}

    g = StateGraph(_S)
    g.add_node("t1_agent", t1_agent, task_ns="t1")
    g.add_node("t2_dummy", lambda s: s, task_ns="t2")
    g.add_node("shared", shared_fn, shared=True)
    g.add_edge(START, "t1_agent")
    g.add_edge("shared", END)

    compiled = g.compile(auto_task_allowlist=True, expand_shared_nodes=True)
    with pytest.raises(InvalidUpdateError, match="denies routing"):
        compiled.invoke(
            {"x": 1},
            {"configurable": {"thread_id": "th", CONFIG_KEY_TASK_NS: "t1"}},
        )


def test_expand_shared_nodes_allows_delegate_to_own_task_proxy() -> None:
    def t1_agent(state: _S):
        return Command(goto=Send("shared__proxy__t1", state))

    def shared_fn(state: _S):
        return {"x": state["x"] + 10}

    g = StateGraph(_S)
    g.add_node("t1_agent", t1_agent, task_ns="t1")
    g.add_node("t2_dummy", lambda s: s, task_ns="t2")
    g.add_node("shared", shared_fn, shared=True)
    g.add_edge(START, "t1_agent")
    g.add_edge("shared", END)

    compiled = g.compile(auto_task_allowlist=True, expand_shared_nodes=True)
    out = compiled.invoke(
        {"x": 1},
        {"configurable": {"thread_id": "th", CONFIG_KEY_TASK_NS: "t1"}},
    )
    assert out["x"] == 11

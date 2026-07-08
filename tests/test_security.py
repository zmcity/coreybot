"""Unit + integration tests for the enterprise security layer.

Covers the four modules under ``coreybot.security`` (secrets, rules, pipeline,
context) and their integration into the agent loop: user info reaches the
system prompt, secrets are redacted on the wire but kept in stored history, the
telemetry stream is scrubbed, and a blocking rule aborts the turn safely.
"""

from __future__ import annotations

import re

import pytest

from coreybot.core.config import Config
from coreybot.core.message import CompletionResult
from coreybot.runtime.agent import Agent
from coreybot.security import (
    Action,
    BlockPatternRule,
    MaxLengthRule,
    OutboundBlocked,
    PatternRedactRule,
    RedactSecretsRule,
    RulePipeline,
    SecretKind,
    SecretStore,
    SecretValue,
    SecurityContext,
    UserInfo,
)
from coreybot.tools import ToolRegistry


# --------------------------------------------------------------------------
# SecretValue / SecretStore
# --------------------------------------------------------------------------
def test_secret_value_never_leaks_via_repr_str_format():
    sv = SecretValue("github_pat_ABCDEFGHIJKLMNOP", name="gh", kind=SecretKind.PAT)
    assert "github_pat_ABCDEFGHIJKLMNOP" not in repr(sv)
    assert "github_pat_ABCDEFGHIJKLMNOP" not in str(sv)
    assert "github_pat_ABCDEFGHIJKLMNOP" not in f"{sv}"
    assert "github_pat_ABCDEFGHIJKLMNOP" not in "x={}".format(sv)
    assert "gh" in repr(sv)  # name is safe to show


def test_secret_value_reveal_returns_raw():
    sv = SecretValue("s3cr3t", name="k")
    assert sv.reveal() == "s3cr3t"
    assert len(sv) == 6
    assert bool(sv) is True
    assert bool(SecretValue("", name="empty")) is False


def test_secret_value_masked_keeps_short_prefix_only():
    sv = SecretValue("github_pat_ABCDEFGHIJKL", name="gh")
    masked = sv.masked(keep=4)
    assert masked.startswith("gith")
    assert "ABCDEFGHIJKL" not in masked
    # keep is clamped to at most a quarter of the length
    assert len(masked) < len("github_pat_ABCDEFGHIJKL")


def test_secret_value_equality_only_against_secret_value():
    a = SecretValue("same", name="a")
    b = SecretValue("same", name="b")
    assert a == b
    # comparing to a plain str is intentionally not equal (NotImplemented -> False)
    assert (a == "same") is False


def test_secret_store_put_get_reveal_and_refs():
    store = SecretStore()
    ref = store.put("tok", "abc123", kind=SecretKind.TOKEN)
    assert ref.name == "tok" and ref.kind == SecretKind.TOKEN
    assert "abc123" not in ref.hint
    assert store.reveal("tok") == "abc123"
    assert "tok" in store
    assert store.names() == ["tok"]
    # refs never expose the value
    assert all("abc123" not in r.hint for r in store.refs())


def test_secret_store_reveal_missing_raises():
    store = SecretStore()
    with pytest.raises(KeyError):
        store.reveal("nope")


def test_secret_store_from_env_reads_and_skips(monkeypatch):
    env = {"A_TOKEN": "v1", "B_TOKEN": "v2"}
    store = SecretStore.from_env(["A_TOKEN", "MISSING", "B_TOKEN"], environ=env)
    assert store.names() == ["A_TOKEN", "B_TOKEN"]
    assert store.reveal("A_TOKEN") == "v1"


def test_secret_store_from_env_required_raises():
    with pytest.raises(KeyError):
        SecretStore.from_env(["MISSING"], environ={}, required=True)


def test_secret_store_repr_lists_names_not_values():
    store = SecretStore()
    store.put("k", "supersecret")
    assert "supersecret" not in repr(store)
    assert "k" in repr(store)


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
def test_redact_secrets_rule_masks_known_value_with_label():
    store = SecretStore()
    store.put("gh_pat", "TOKENVALUE123456", kind=SecretKind.PAT)
    rule = RedactSecretsRule(store)
    res = rule.apply("use TOKENVALUE123456 now")
    assert res.action == Action.REDACT
    assert "TOKENVALUE123456" not in res.text
    assert "[REDACTED:gh_pat]" in res.text
    assert res.hits == 1


def test_redact_secrets_rule_allow_when_absent():
    store = SecretStore()
    store.put("k", "SECRET")
    res = RedactSecretsRule(store).apply("nothing here")
    assert res.action == Action.ALLOW
    assert res.text == "nothing here"


def test_redact_secrets_rule_longer_first():
    store = SecretStore()
    store.put("short", "ABCDEF")
    store.put("long", "ABCDEF-XYZ-1234567890")
    text = "value ABCDEF-XYZ-1234567890 end"
    res = RedactSecretsRule(store).apply(text)
    # the longer secret should be masked as a whole, not partially by the short one
    assert "[REDACTED:long]" in res.text
    assert "ABCDEF-XYZ-1234567890" not in res.text


@pytest.mark.parametrize(
    "token",
    [
        "ghp_ABCDEFGHIJKLMNOPQRSTUVWX123456",
        "github_pat_11ABCDEFG_ABCDEFGHIJKLMNOPQRST",
        "sk-ABCDEFGHIJKLMNOPQRSTUVWX",
        "AKIAIOSFODNN7EXAMPLE",
    ],
)
def test_pattern_redact_rule_masks_token_shapes(token):
    res = PatternRedactRule().apply(f"secret={token} trailing")
    assert res.action == Action.REDACT
    assert token not in res.text


def test_pattern_redact_rule_masks_bearer_header():
    res = PatternRedactRule().apply("Authorization: Bearer abcdefghijklmnop123456")
    assert "abcdefghijklmnop123456" not in res.text


def test_pattern_redact_rule_allow_plain_text():
    res = PatternRedactRule().apply("just a normal sentence with numbers 12345")
    assert res.action == Action.ALLOW


def test_block_pattern_rule_blocks_match():
    rule = BlockPatternRule([re.compile(r"CLASSIFIED")])
    res = rule.apply("this is CLASSIFIED material")
    assert res.action == Action.BLOCK
    assert "CLASSIFIED" in res.reason


def test_block_pattern_rule_requires_patterns():
    with pytest.raises(ValueError):
        BlockPatternRule([])


def test_max_length_rule_truncates_by_default():
    rule = MaxLengthRule(10)
    res = rule.apply("x" * 50)
    assert res.action == Action.REDACT
    assert res.text.startswith("x" * 10)
    assert "truncated" in res.text


def test_max_length_rule_can_block():
    rule = MaxLengthRule(10, block=True)
    res = rule.apply("x" * 50)
    assert res.action == Action.BLOCK


def test_max_length_rule_allow_when_short():
    assert MaxLengthRule(10).apply("short").action == Action.ALLOW


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def test_pipeline_cumulative_redaction():
    store = SecretStore()
    store.put("k", "KNOWNSECRET")
    pipe = RulePipeline([RedactSecretsRule(store), PatternRedactRule()])
    outcome = pipe.run("known KNOWNSECRET and token ghp_ABCDEFGHIJKLMNOPQRSTUV12345")
    assert outcome.changed
    assert "KNOWNSECRET" not in outcome.text
    assert "ghp_ABCDEFGHIJKLMNOPQRSTUV12345" not in outcome.text
    assert outcome.total_hits >= 2
    assert "redacted" in outcome.redaction_summary()


def test_pipeline_block_raises_outbound_blocked():
    pipe = RulePipeline([BlockPatternRule([re.compile(r"STOP")])])
    with pytest.raises(OutboundBlocked) as exc:
        pipe.run("please STOP now")
    assert exc.value.rule == "block_patterns"


def test_pipeline_allow_leaves_text_untouched_and_empty_trail():
    pipe = RulePipeline([PatternRedactRule()])
    outcome = pipe.run("nothing sensitive")
    assert outcome.text == "nothing sensitive"
    assert outcome.changed is False
    assert outcome.trail == []


def test_pipeline_run_text_convenience():
    store = SecretStore()
    store.put("k", "ZZZ")
    pipe = RulePipeline([RedactSecretsRule(store)])
    assert "ZZZ" not in pipe.run_text("value ZZZ")


def test_pipeline_add_is_chainable():
    pipe = RulePipeline()
    assert pipe.add(PatternRedactRule()) is pipe
    assert len(pipe) == 1


# --------------------------------------------------------------------------
# UserInfo / SecurityContext
# --------------------------------------------------------------------------
def test_user_info_render_block_contains_fields():
    info = UserInfo(user_id="u1", display_name="Ada", roles=["admin", "dev"],
                    attributes={"team": "core"})
    block = info.render_block()
    assert "<user_context>" in block
    assert "u1" in block and "Ada" in block
    assert "admin, dev" in block
    assert "team" in block and "core" in block


def test_user_info_empty_renders_nothing():
    assert UserInfo().render_block() == ""
    assert UserInfo().is_empty()


def test_security_context_default_pipeline_protects_out_of_the_box():
    ctx = SecurityContext.for_user(user_id="u1")
    ctx.add_secret("pat", "github_pat_ABCDEFGHIJKLMNOPQRST", kind=SecretKind.PAT)
    out = ctx.vet_outbound("token github_pat_ABCDEFGHIJKLMNOPQRST here")
    assert "github_pat_ABCDEFGHIJKLMNOPQRST" not in out.text


def test_security_context_redact_scrubs_without_policy():
    ctx = SecurityContext.for_user()
    ctx.add_secret("k", "SECRETXYZ")
    assert "SECRETXYZ" not in ctx.redact("leak SECRETXYZ end")


def test_security_context_redact_can_be_disabled():
    store = SecretStore()
    store.put("k", "SECRETXYZ")
    ctx = SecurityContext(secrets=store, auto_redact_telemetry=False)
    # disabled -> passthrough
    assert ctx.redact("leak SECRETXYZ") == "leak SECRETXYZ"


def test_security_context_repr_hides_secrets():
    ctx = SecurityContext.for_user(user_id="u1")
    ctx.add_secret("k", "SUPERSECRET")
    assert "SUPERSECRET" not in repr(ctx)
    assert "u1" in repr(ctx)


def test_security_context_add_rule_chainable():
    ctx = SecurityContext.for_user()
    assert ctx.add_rule(MaxLengthRule(100)) is ctx


# --------------------------------------------------------------------------
# Agent integration
# --------------------------------------------------------------------------
class _CaptureProvider:
    """Records the messages it receives; returns one scripted message reply."""

    def __init__(self, reply="<message>ok</message>"):
        self.reply = reply
        self.seen = None
        self.calls = 0

    async def acomplete(self, messages, cancel_token=None):
        self.calls += 1
        self.seen = [(m.role.value, m.content) for m in messages]
        return CompletionResult(text=self.reply, model="fake")


def _wire_blob(provider):
    return "\n".join(content for _, content in provider.seen)


@pytest.mark.asyncio
async def test_agent_injects_user_block_into_system_prompt():
    sec = SecurityContext.for_user(user_id="u42", display_name="Ada", roles=["admin"])
    agent = Agent(Config(system_prompt="Be nice."), provider=_CaptureProvider(),
                  registry=ToolRegistry(), security=sec)
    assert "<user_context>" in agent.system_prompt
    assert "u42" in agent.system_prompt


@pytest.mark.asyncio
async def test_agent_redacts_secret_on_wire_but_keeps_history_and_scrubs_telemetry():
    sec = SecurityContext.for_user(user_id="u1")
    sec.add_secret("gh_pat", "github_pat_LEAKME1234567890ABCD", kind=SecretKind.PAT)
    provider = _CaptureProvider()
    agent = Agent(Config(system_prompt="Be nice."), provider=provider,
                  registry=ToolRegistry(), security=sec)

    await agent.arun_turn("my token github_pat_LEAKME1234567890ABCD store it")

    wire = _wire_blob(provider)
    assert "github_pat_LEAKME1234567890ABCD" not in wire       # not sent
    assert "[REDACTED:gh_pat]" in wire                          # masked
    stored = "\n".join(m.content for m in agent.history)
    assert "github_pat_LEAKME1234567890ABCD" in stored          # kept for audit
    tele = "\n".join(e.text for e in agent.telemetry if e.kind == "llm_call")
    assert "github_pat_LEAKME1234567890ABCD" not in tele        # telemetry scrubbed


@pytest.mark.asyncio
async def test_agent_without_security_is_unchanged():
    provider = _CaptureProvider()
    agent = Agent(Config(system_prompt="Be nice."), provider=provider,
                  registry=ToolRegistry())
    await agent.arun_turn("hello github_pat_UNTOUCHED1234567890AB")
    # no security -> content is sent verbatim (identity path)
    assert "github_pat_UNTOUCHED1234567890AB" in _wire_blob(provider)
    assert "<user_context>" not in agent.system_prompt


@pytest.mark.asyncio
async def test_agent_outbound_block_aborts_turn_safely():
    sec = SecurityContext.for_user(user_id="u1")
    sec.add_rule(BlockPatternRule([re.compile(r"NUCLEAR")]))
    provider = _CaptureProvider()
    agent = Agent(Config(system_prompt="Be nice."), provider=provider,
                  registry=ToolRegistry(), security=sec)

    resp = await agent.arun_turn("launch the NUCLEAR codes")
    # provider must never have been called
    assert provider.calls == 0
    assert "blocked" in resp.content.lower()
    # a notice event explains the block
    notices = [e for e in agent.telemetry if e.kind == "notice"]
    assert any("blocked" in e.text.lower() for e in notices)

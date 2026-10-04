"""Opt-in cross-project verification against an explicit repaired Hermes checkout.

Run with Hermes's Python: tests/check_topic_policy_integration.py /path/to/hermes-agent
Only temporary copies are patched. SDK transports and agent execution are mocked.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest

# An explicit checkout is mandatory; never discover or patch an installed gateway.
ROOT = Path(sys.argv[1]).resolve()
PLUGIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PLUGIN), str(ROOT)]

from gateway.config import Platform, PlatformConfig  # noqa: E402
from gateway.platforms.base import BasePlatformAdapter, _thread_metadata_for_event  # noqa: E402
from plugins.platforms.feishu import adapter as core_module  # noqa: E402
from plugins.platforms.feishu import adapter_delivery  # noqa: E402

assert Path(core_module.__file__).resolve() == ROOT / "plugins/platforms/feishu/adapter.py"
assert core_module._load_lark_oapi(), "The real Lark SDK is required"

from lark_oapi.api.im.v1 import CreateMessageRequest, ListMessageRequest, ReplyMessageRequest  # noqa: E402
from test_split_gateway import method  # noqa: E402
from test_topic_delivery import (  # noqa: E402
    bad,
    controller,
    event_for,
    generated_complete,
    generated_start,
    ok,
)

from hermes_lark_streaming.patcher import CronPatcher, Patcher, install_patchers  # noqa: E402
from hermes_lark_streaming.split_gateway import GATEWAY_FILES  # noqa: E402

logging.disable(logging.CRITICAL)


def git_identity():
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD", "HEAD^{tree}"], text=True,
    ).splitlines()


def checksum(paths):
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def core_adapter(policy, scenario):
    core = core_module.FeishuAdapter(PlatformConfig(extra={"topic_delivery_fallback": policy}))
    core._client = Mock()
    messages = core._client.im.v1.message
    messages.create.return_value = ok(message_id="om_core_parent")
    messages.reply.return_value = bad(230011)
    messages.list.return_value = ok(items=[])
    if scenario == "valid":
        messages.reply.return_value = ok(message_id="om_core_topic")
    elif scenario == "recovered":
        messages.list.return_value = ok(items=[NS(message_id="om_recovered", thread_id="omt_topic")])
        messages.reply.side_effect = lambda request: (
            ok(message_id="om_core_topic") if request.message_id == "om_recovered" else bad(230011)
        )
    elif scenario == "lookup_failed":
        messages.list.return_value = bad(99991663)
    elif scenario == "stale231003":
        messages.reply.return_value = bad(231003)
    elif scenario == "permission":
        messages.reply.return_value = bad(99991663)
    elif scenario == "rate_limit":
        messages.reply.return_value = bad(230020)
    elif scenario == "timeout":
        messages.reply.side_effect = TimeoutError("Synthetic ambiguous send")
    return core


def metadata(event):
    result = _thread_metadata_for_event(event) or {}
    result["reply_to_message_id"] = getattr(event, "reply_anchor_override", None) or event.anchor
    return result


async def native_final(generated, core, event, result, text):
    owner = NS(
        _event_thread_metadata=lambda *args: metadata(event),
        _delivery_adapter_for=lambda _: core,
        _should_send_voice_reply=lambda *a, **kw: False,
        _deliver_media_from_response=AsyncMock(),
    )
    owner._turn_retry_suppressed_result = method(
        generated, "run_turn.py", "_turn_retry_suppressed_result",
    )
    owner._mark_delivery_retry_suppressed = method(
        generated, "run_turn.py", "_mark_delivery_retry_suppressed",
    )
    deliver = method(generated, "run_turn.py", "_hmwa_deliver_turn_response", diagnostic_wake_muted=lambda _: False)
    response = await deliver(owner, event, event.source, NS(session_id="session"), "session", 1,
                             result, [], text, "", False)
    if response is not None:
        return await core.send(event.source.chat_id, response,
                               reply_to=metadata(event)["reply_to_message_id"], metadata=metadata(event))
    return None


async def native_background(generated, core, event, mp, attachments):
    # Model only agent execution and external media I/O. The complete patched
    # background method and real native adapter routing execute unchanged.
    mp.setitem(sys.modules, "gateway.run", NS(
        _checkpoint_agent_kwargs=Mock(), _current_max_iterations=lambda: 10,
        _load_gateway_config=lambda: {}, _platform_config_key=lambda _: "feishu",
    ))
    mp.setitem(sys.modules, "run_agent", NS(AIAgent=Mock()))
    mp.setitem(sys.modules, "gateway.run_notifications", NS(_IMAGE_EXTS=set(), _VIDEO_EXTS=set()))
    core.extract_media = lambda _: ([], "ORIGINAL_BODY")
    core.extract_images = lambda _: (
        [("https://invalid.example/image", "caption")] if attachments else [], "ORIGINAL_BODY",
    )

    async def image_send(**kwargs):
        return await core.send(kwargs["chat_id"], "ORIGINAL_IMAGE_CAPTION", metadata=kwargs["metadata"])

    core.send_image = AsyncMock(side_effect=image_send)
    owner = NS(
        _delivery_adapter_for=lambda _: core,
        _thread_metadata_for_source=lambda *a: metadata(event),
        _resolve_session_agent_runtime=lambda **k: ("model", {"api_key": "test"}),
        _resolve_turn_toolsets=lambda *a: ([], []), _provider_routing={},
        _resolve_session_reasoning_config=lambda **k: {}, _resolve_session_service_tier=lambda **k: None,
        _resolve_turn_agent_config=lambda *a: {},
        _run_in_executor_with_context=AsyncMock(return_value={"final_response": "ORIGINAL_BODY"}),
    )
    fn = method(generated, "run_turn.py", "_run_background_task_inner", os=os, Platform=Platform,
                t=lambda key, **kw: "Background: " if "header" in key else key,
                repair_explicit_computer_use_media_paths=lambda text, _: text,
                BasePlatformAdapter=BasePlatformAdapter)
    await fn(owner, "ORIGINAL_PROMPT", event.source, "task", event_message_id=event.anchor)


async def run_matrix(generated, home):
    results = []
    for policy in ("main_chat", "error_notice", "silent"):
        for lane in ("background", "normal", "status", "redirect"):
            scenarios = ("missing", "stale", "stale231003", "recovered", "lookup_failed",
                         "valid", "permission", "rate_limit", "timeout")
            if lane == "normal":
                scenarios += ("content_rejection",)
            if lane == "status":
                scenarios = ("missing",)
            elif lane == "redirect":
                scenarios = ("missing", "stale", "valid")
            for scenario in scenarios:
                with pytest.MonkeyPatch.context() as mp:
                    ctrl, client = controller(home, mp)
                    mp.setattr(adapter_delivery, "asyncio", NS(**{**vars(asyncio), "sleep": AsyncMock()}))
                    event = event_for(anchor=None if scenario == "missing" else "om_anchor")
                    event.source.platform = Platform.FEISHU
                    event.reply_to_message_id = event.anchor
                    core = core_adapter(policy, scenario)
                    if lane == "background":
                        await native_background(generated, core, event, mp, attachments=True)
                    else:
                        if lane == "status":
                            await core.send(event.source.chat_id, "STATUS_BODY", metadata=metadata(event))
                        if scenario != "valid":
                            client._client.im.v1.message.areply.return_value = bad(
                                230099 if scenario == "content_rejection" else 230011,
                            )
                        if lane == "redirect":
                            client._client.im.v1.message.areply.return_value = ok(message_id="om_initial_card")
                        generated_start(event)
                        await ctrl._sessions[event.message_id].create_task
                        if lane == "redirect":
                            incoming = event_for(anchor="om_redirect_anchor")
                            incoming.message_id = "om_redirect"
                            incoming.source.platform = Platform.FEISHU
                            incoming.reply_to_message_id = incoming.anchor
                            context = NS(stream_consumer_holder=[None], _progress_metadata={},
                                         _status_thread_metadata={})
                            turn = NS(event=event, ctx=context)
                            owner = NS(_try_agent_verb=lambda *a, **kw: True,
                                       _fold_into_running_turn=lambda *a, current_turn=turn: current_turn,
                                       _reply_anchor_for_event=lambda e: e.anchor,
                                       _event_thread_metadata=lambda e, s: metadata(e))
                            redirect = method(generated, "run_busy.py", "_redirect_active_turn", Platform=Platform)
                            assert redirect(owner, object(), "redirect", "session", incoming)
                            core._client.im.v1.message.reply.return_value = ok(message_id="om_redirect_reply")
                        result, text = await generated_complete(event)
                        await native_final(generated, core, event, result, text)
                    plugin_messages = client._client.im.v1.message
                    messages = core._client.im.v1.message
                    for calls, request_type in (
                        (messages.create.call_args_list, CreateMessageRequest),
                        (messages.reply.call_args_list, ReplyMessageRequest),
                        (messages.list.call_args_list, ListMessageRequest),
                        (plugin_messages.acreate.await_args_list, CreateMessageRequest),
                        (plugin_messages.areply.await_args_list, ReplyMessageRequest),
                    ):
                        assert all(isinstance(call.args[0], request_type) for call in calls)
                    parent_payloads = [call.args[0].request_body.content for call in messages.create.call_args_list]
                    assert plugin_messages.acreate.await_count == 0, (policy, lane, scenario, "plugin parent leak")
                    if policy in ("error_notice", "silent"):
                        assert not any("ORIGINAL_" in payload or "STATUS_BODY" in payload
                                       for payload in parent_payloads)
                        assert len(parent_payloads) <= (1 if policy == "error_notice" else 0)
                    exhausted = scenario in ("missing", "stale", "stale231003", "lookup_failed", "content_rejection")
                    if exhausted and lane != "redirect":
                        expected = 0 if policy == "silent" else 1
                        if policy == "main_chat" and lane in ("background", "status"):
                            expected = 2
                        assert len(parent_payloads) == expected, (policy, lane, scenario, parent_payloads)
                        if policy == "main_chat":
                            assert any("ORIGINAL_BODY" in payload for payload in parent_payloads)
                    if scenario in ("permission", "rate_limit", "timeout"):
                        assert not parent_payloads
                        assert messages.list.call_count == 0
                    if scenario == "recovered":
                        assert not parent_payloads
                        assert any(call.args[0].message_id == "om_recovered" for call in messages.reply.call_args_list)
                    if lane == "normal" and scenario == "valid":
                        assert client.cardkit_update.await_count == 1
                        assert not messages.create.called and not messages.reply.called
                    if lane == "redirect":
                        assert client.cardkit_update.await_count == 0
                        assert messages.reply.call_count == 1
                        assert messages.reply.call_args.args[0].message_id == "om_redirect_anchor"
                    results.append({"policy": policy, "lane": lane, "scenario": scenario,
                                    "plugin_parent": plugin_messages.acreate.await_count,
                                    "native_parent": messages.create.call_count,
                                    "native_reply": messages.reply.call_count,
                                    "history": messages.list.call_count})
                    executor = getattr(core, "_sdk_executor", None)
                    if executor:
                        executor.shutdown(wait=True)
    return results


def main():
    before_identity = git_identity()
    files = [ROOT / path for path in (
        "gateway/run.py", *(f"gateway/{name}" for name in GATEWAY_FILES),
        "cron/scheduler.py", "cron/scheduler_delivery.py",
    )]
    before = checksum(files)
    with tempfile.TemporaryDirectory(prefix="lark-topic-integration-") as directory:
        target = Path(directory)
        for file in files:
            dest = target / file.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest)
        patchers = [Patcher(target / "gateway/run.py"), CronPatcher(target / "cron/scheduler.py")]
        for patcher in patchers:
            patcher.verify_target()
        install_patchers(patchers)
        assert all(patcher.is_fully_patched() for patcher in patchers)
        installed = {path: path.read_bytes() for path in target.rglob("*.py")}
        install_patchers(patchers)
        assert all(path.read_bytes() == content for path, content in installed.items())
        generated = {name: (target / "gateway" / name).read_text() for name in GATEWAY_FILES}
        results = asyncio.run(run_matrix(generated, target))
        for patcher in patchers:
            patcher.remove()
        assert all((target / file.relative_to(ROOT)).read_bytes() == file.read_bytes() for file in files)
    assert git_identity() == before_identity
    assert checksum(files) == before
    print(json.dumps({"core_commit": before_identity[0], "core_tree": before_identity[1],
                      "roundtrip": "byte-identical", "sdk": "real lark-oapi, explicitly loaded",
                      "cases": results}, indent=2))


if __name__ == "__main__":
    main()

"""Topic routing stays native while valid CardKit replies keep streaming."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest

import hermes_lark_streaming.feishu as feishu_module
import hermes_lark_streaming.patch as hooks
from hermes_lark_streaming.controller import StreamCardController
from hermes_lark_streaming.feishu import FeishuAPIError, FeishuClient, topic_delivery_scope
from hermes_lark_streaming.patcher import _bg_deliver_hook, _complete_body, _start_hook
from hermes_lark_streaming.streaming.session import SessionState


def ok(**fields):
    return NS(success=lambda: True, data=NS(**fields))


def bad(code):
    return NS(success=lambda: False, code=code, msg="Synthetic rejection")


def controller(home, monkeypatch):
    ctrl = StreamCardController(profile_home=home)
    ctrl._cfg._raw = {
        "streaming": {"enabled": True, "footer": {"enabled": False}},
        "feishu": {"app_id": "test-app", "app_secret": "test-secret"},
    }
    ctrl._initialized = True
    client = object.__new__(FeishuClient)
    client._client = NS(im=NS(v1=NS(message=NS(
        acreate=AsyncMock(return_value=ok(message_id="om_plugin_parent")),
        areply=AsyncMock(return_value=ok(message_id="om_plugin_reply")),
    ))))
    client.cardkit_create = AsyncMock(return_value="card_mock")
    client.cardkit_close_streaming = AsyncMock()
    client.cardkit_update = AsyncMock()
    ctrl._client = client
    monkeypatch.setattr(hooks, "get_controller", lambda: ctrl)
    return ctrl, client


def event_for(thread_id="omt_topic", anchor="om_anchor"):
    source = NS(platform=NS(value="feishu"), chat_id="oc_origin", thread_id=thread_id)
    return NS(message_id="om_inbound", source=source, anchor=anchor)


def generated_start(event):
    namespace = {}
    exec("def start(self, event, source):\n" + _start_hook("    "), namespace)
    namespace["start"](NS(_reply_anchor_for_event=lambda e: e.anchor), event, event.source)


async def generated_complete(event, answer="ORIGINAL_BODY", *, result=None):
    namespace = {}
    source = '''async def complete(event, answer, agent_result):
    response = _lark_completion_answer = _lark_original_response = answer
    _turn_seconds = 1
    _footer_line = ""
''' + _complete_body("    ") + '''
    return agent_result, response
'''
    exec(source, namespace)
    return await namespace["complete"](event, answer, {"already_sent": True} if result is None else result)


async def generated_background(event, *, attachments=False):
    namespace = {"logger": logging.getLogger(__name__)}
    source = '''async def background(source, event_message_id, attachments):
    response = text_content = "ORIGINAL_BODY"
    prompt = "ORIGINAL_PROMPT"
    images = ["original-image"] if attachments else []
    media_files = ["original-file"] if attachments else []
''' + _bg_deliver_hook("    ") + '''
    return text_content, images, media_files
'''
    exec(source, namespace)
    return await namespace["background"](event.source, event.anchor, attachments)


@pytest.mark.asyncio
@pytest.mark.parametrize("anchor", [None, "om_stale", "om_valid"])
@pytest.mark.parametrize("attachments", [False, True])
async def test_topic_background_yields_without_touching_content(tmp_path, monkeypatch, anchor, attachments):
    _ctrl, client = controller(tmp_path, monkeypatch)
    event = event_for(anchor=anchor)
    assert await generated_background(event, attachments=attachments) == (
        "ORIGINAL_BODY", ["original-image"] if attachments else [], ["original-file"] if attachments else [],
    )
    client._client.im.v1.message.acreate.assert_not_awaited()
    client._client.im.v1.message.areply.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [230099, 230011, 231003, 99992402, 99991663, 230020])
async def test_failed_topic_card_yields_full_native_completion(tmp_path, monkeypatch, failure):
    ctrl, client = controller(tmp_path, monkeypatch)
    client._client.im.v1.message.areply.return_value = bad(failure)
    event = event_for()
    generated_start(event)
    await ctrl._sessions[event.message_id].create_task
    result, response = await generated_complete(event)
    assert not result.get("already_sent")
    assert response == "ORIGINAL_BODY"
    client._client.im.v1.message.acreate.assert_not_awaited()
    client.cardkit_update.assert_not_awaited()
    replies = client._client.im.v1.message.areply.await_args_list
    assert len(replies) == (2 if failure == 230099 else 1)
    assert all(call.args[0].request_body.reply_in_thread is True for call in replies)


@pytest.mark.asyncio
async def test_valid_topic_stream_and_rollover_stay_in_thread(tmp_path, monkeypatch):
    ctrl, client = controller(tmp_path, monkeypatch)
    event = event_for()
    generated_start(event)
    session = ctrl._sessions[event.message_id]
    await session.create_task
    assert session.thread_id == event.source.thread_id
    assert session.state == SessionState.STREAMING
    assert await ctrl._create_streaming_card(session) == ("card_mock", "om_plugin_reply")
    result, _ = await generated_complete(event)
    assert result["already_sent"] is True
    for call in client._client.im.v1.message.areply.await_args_list:
        assert call.args[0].message_id == event.anchor
        assert call.args[0].request_body.reply_in_thread is True
    assert "ORIGINAL_BODY" in json.dumps(client.cardkit_update.await_args.args[1])
    client._client.im.v1.message.acreate.assert_not_awaited()


@pytest.mark.asyncio
async def test_flat_chat_content_rejection_keeps_existing_parent_fallback(tmp_path, monkeypatch):
    ctrl, client = controller(tmp_path, monkeypatch)
    client._client.im.v1.message.areply.return_value = bad(230099)
    event = event_for(thread_id=None)
    generated_start(event)
    await ctrl._sessions[event.message_id].create_task
    result, _ = await generated_complete(event)
    assert result["already_sent"] is True
    client._client.im.v1.message.acreate.assert_awaited_once()
    assert all(call.args[0].request_body.reply_in_thread is None
               for call in client._client.im.v1.message.areply.await_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("anchor", [None, "om_quote"])
async def test_flat_background_keeps_card_delivery(tmp_path, monkeypatch, anchor):
    _ctrl, client = controller(tmp_path, monkeypatch)
    assert await generated_background(event_for(thread_id=None, anchor=anchor)) is None
    messages = client._client.im.v1.message
    assert messages.areply.await_count == bool(anchor)
    assert messages.acreate.await_count == (not anchor)


@pytest.mark.parametrize("raw", [None, {}, {"synthetic": True}, {"root_id": "om_other"}])
def test_unknown_raw_payload_preserves_authoritative_topic(tmp_path, monkeypatch, raw):
    controller(tmp_path, monkeypatch)
    event = event_for()
    event.reply_to_message_id = "om_quote"
    event.raw_message = raw
    hooks.on_feishu_normalize(message_id=event.message_id, event=event, source=event.source)
    assert event.source.thread_id == "omt_topic"


@pytest.mark.parametrize("native_thread", [None, "omt_topic"])
def test_normalize_repairs_only_proven_legacy_quote(tmp_path, monkeypatch, native_thread):
    controller(tmp_path, monkeypatch)
    event = event_for(thread_id=native_thread or "om_quote")
    event.reply_to_message_id = "om_quote"
    event.raw_message = {"event": {"message": {"root_id": "om_quote", "thread_id": native_thread}}}
    hooks.on_feishu_normalize(message_id=event.message_id, event=event, source=event.source)
    assert event.source.thread_id == native_thread


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["terminal", "parent_chat", "home", "recovered", "redirect"])
@pytest.mark.parametrize("before_creation", [False, True])
async def test_native_routing_change_yields_cards(tmp_path, monkeypatch, changed, before_creation):
    ctrl, client = controller(tmp_path, monkeypatch)
    event = event_for()
    generated_start(event)
    session = ctrl._sessions[event.message_id]
    if not before_creation:
        await session.create_task
    if changed == "terminal":
        event._feishu_topic_delivery = {"terminal": NS(retry_suppressed=True)}
    elif changed in {"parent_chat", "home"}:
        event._feishu_topic_delivery = {"destination": changed}
    elif changed == "recovered":
        event._feishu_topic_delivery = {"anchor": "om_recovered"}
    else:
        event.ledger_message_id = "om_redirect"
        event.reply_anchor_override = "om_redirect_anchor"
    await session.create_task
    if before_creation:
        client.cardkit_create.assert_not_awaited()
    else:
        before = client._client.im.v1.message.areply.await_count
        assert await ctrl._create_streaming_card(session) is None
        assert client._client.im.v1.message.areply.await_count == before
    result, response = await generated_complete(event)
    assert not result.get("already_sent")
    assert response == "ORIGINAL_BODY"
    client.cardkit_update.assert_not_awaited()
    client._client.im.v1.message.acreate.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_terminal_during_batch_stops_subsequent_stream_elements(tmp_path, monkeypatch):
    ctrl, client = controller(tmp_path, monkeypatch)
    event = event_for()
    generated_start(event)
    session = ctrl._sessions[event.message_id]
    await session.create_task
    session.segment_state.on_answer_delta("BODY_BUFFERED_BEFORE_NATIVE_STOP")

    async def stop_during_batch(*_args, **_kwargs):
        event._feishu_topic_delivery = {"terminal": NS(retry_suppressed=True)}

    client.cardkit_batch_update = AsyncMock(side_effect=stop_during_batch)
    client.cardkit_stream_element = AsyncMock()
    await ctrl._do_flush(session)

    client.cardkit_batch_update.assert_awaited_once()
    client.cardkit_stream_element.assert_not_awaited()
    result, response = await generated_complete(event)
    assert not result.get("already_sent")
    assert response == "ORIGINAL_BODY"
    client.cardkit_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_terminal_during_seal_close_stops_full_card_update(tmp_path, monkeypatch):
    ctrl, client = controller(tmp_path, monkeypatch)
    event = event_for()
    generated_start(event)
    session = ctrl._sessions[event.message_id]
    await session.create_task
    session.segment_state.on_answer_delta("SEALED_BODY_BUFFERED_BEFORE_NATIVE_STOP")

    async def stop_during_close(*_args, **_kwargs):
        event._feishu_topic_delivery = {"terminal": NS(retry_suppressed=True)}

    client.cardkit_close_streaming = AsyncMock(side_effect=stop_during_close)
    await ctrl._seal_current_card(session, session.active_segments())

    client.cardkit_close_streaming.assert_awaited_once()
    client.cardkit_update.assert_not_awaited()
    result, response = await generated_complete(event)
    assert not result.get("already_sent")
    assert response == "ORIGINAL_BODY"


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["terminal", "parent_chat", "home", "redirect"])
async def test_native_change_during_final_update_does_not_claim_delivery(tmp_path, monkeypatch, changed):
    ctrl, client = controller(tmp_path, monkeypatch)
    event = event_for()
    generated_start(event)
    await ctrl._sessions[event.message_id].create_task

    async def change_during_update(*_args, **_kwargs):
        if changed == "terminal":
            event._feishu_topic_delivery = {"terminal": NS(retry_suppressed=True)}
        elif changed in {"parent_chat", "home"}:
            event._feishu_topic_delivery = {"destination": changed}
        else:
            event.ledger_message_id = "om_redirect"
            event.reply_anchor_override = "om_redirect_anchor"

    client.cardkit_update = AsyncMock(side_effect=change_during_update)
    result, response = await generated_complete(event)

    client.cardkit_update.assert_awaited_once()
    assert not result.get("already_sent")
    assert response == "ORIGINAL_BODY"
    assert not ctrl._sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["terminal", "parent_chat", "home"])
async def test_topic_retry_rechecks_native_ownership_and_restores_scope(tmp_path, monkeypatch, changed):
    _ctrl, client = controller(tmp_path, monkeypatch)
    event = event_for()
    replies = client._client.im.v1.message.areply
    replies.side_effect = [bad(2200), ok(message_id="om_later_flat_reply")]

    async def stop_during_retry_delay(_delay):
        event._feishu_topic_delivery = (
            {"terminal": NS(retry_suppressed=True)} if changed == "terminal" else {"destination": changed}
        )

    sleep = AsyncMock(side_effect=stop_during_retry_delay)
    monkeypatch.setattr(feishu_module.asyncio, "sleep", sleep)
    guard = hooks._topic_delivery_guard(event, event.source.thread_id, event.anchor)
    with topic_delivery_scope(guard), pytest.raises(FeishuAPIError):
        await client.reply_card_by_id(event.anchor, "card", reply_in_thread=True)

    replies.assert_awaited_once()
    sleep.assert_awaited_once()
    # The aborted topic's guard must not leak into an unrelated later flat send.
    assert await client.reply_card_by_id("om_flat_anchor", "card") == "om_later_flat_reply"
    assert replies.await_count == 2


@pytest.mark.asyncio
async def test_created_topic_task_scopes_transient_reply_retries(tmp_path, monkeypatch):
    ctrl, client = controller(tmp_path, monkeypatch)
    event = event_for()
    replies = client._client.im.v1.message.areply
    replies.side_effect = [bad(2200), ok(message_id="om_unwanted_retry")]

    async def redirect_during_retry_delay(_delay):
        event.ledger_message_id = "om_redirect"
        event.reply_anchor_override = "om_redirect_anchor"

    monkeypatch.setattr(feishu_module.asyncio, "sleep", AsyncMock(side_effect=redirect_during_retry_delay))
    generated_start(event)
    await ctrl._sessions[event.message_id].create_task
    result, response = await generated_complete(event)

    replies.assert_awaited_once()
    assert not result.get("already_sent")
    assert response == "ORIGINAL_BODY"
    client._client.im.v1.message.acreate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("media_delivered", [False, True])
async def test_redirected_outer_completion_preserves_native_confirmed_first_response(
    tmp_path, monkeypatch, media_delivered,
):
    ctrl, client = controller(tmp_path, monkeypatch)
    event = event_for()
    generated_start(event)
    await ctrl._sessions[event.message_id].create_task
    event.ledger_message_id = "om_redirect"
    event.reply_anchor_override = "om_redirect_anchor"
    # Current Hermes sets this key only after its queued first-response text was
    # delivered. A rejected child preparation then returns that same result to
    # the outer completion, whose old CardSession must not undo native delivery.
    result = {"already_sent": True, "media_already_delivered": media_delivered}
    result, response = await generated_complete(event, result=result)

    assert result["already_sent"] is True
    assert result["media_already_delivered"] is media_delivered
    assert response == "ORIGINAL_BODY"
    assert not ctrl._sessions
    assert not ctrl.consume_text_fallback(event.message_id)
    client.cardkit_update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupted", [False, True])
async def test_generated_followup_carries_own_topic_and_event(monkeypatch, interrupted):
    from hermes_sources import patched_gateway
    from test_split_gateway import _stub_gateway_packages, context, method

    _stub_gateway_packages(monkeypatch)
    started = Mock()
    interrupted_hook = Mock()
    monkeypatch.setattr(hooks, "on_message_started", started)
    monkeypatch.setattr(hooks, "on_message_interrupted", interrupted_hook)
    monkeypatch.setattr(hooks, "on_queued_followup_result", Mock())
    fn = method({"run_turn.py": patched_gateway("run_turn.py")}, "run_turn.py", "_run_agent_queued_followup")
    prior = event_for(thread_id="omt_original", anchor="om_original_anchor")
    pending = event_for(thread_id="omt_next", anchor="om_next_anchor")
    pending.message_id = "om_next_inbound"
    ctx = context(source=prior.source, inbound_message_id=prior.message_id,
                  event_message_id=prior.anchor, session_id="sid", history=[],
                  _interrupt_depth=0, context_prompt="prompt")
    owner = NS(_MAX_INTERRUPT_DEPTH=10, _is_goal_continuation_event=lambda _: False,
               _session_key_for_source=lambda _: "next-session", _reply_anchor_for_event=lambda e: e.anchor,
               _prepare_profile_scoped_inbound_message_text=AsyncMock(return_value="next"),
               _adapter_for_source=lambda _: None, _refresh_agent_cache_message_count=AsyncMock(),
               _run_agent_deliver_first_response=AsyncMock(),
               _run_agent=AsyncMock(return_value={"final_response": "next answer"}))

    await fn(owner, ctx, None, "next", pending, {}, {"messages": [], "interrupted": interrupted}, None)

    called, unused = (interrupted_hook, started) if interrupted else (started, interrupted_hook)
    called.assert_called_once()
    unused.assert_not_called()
    values = called.call_args.kwargs
    assert values["thread_id"] == pending.source.thread_id
    assert values["event"] is pending
    assert values["anchor_id"] == pending.anchor
    assert values["chat_id"] == pending.source.chat_id
    assert values["new_message_id" if interrupted else "message_id"] == pending.message_id

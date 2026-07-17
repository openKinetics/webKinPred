import logging
import os
import signal
import time
from typing import Any

import redis
from django.conf import settings

redis_conn: Any = redis.from_url(settings.LOGGING_REDIS_URL, decode_responses=True)

_LOG_TTL = 3600  # seconds — list persists for 1 h after session ends
_log = logging.getLogger(__name__)


def get_channel_name(session_id: str) -> str:
    return f"session_logs:{session_id}"


def get_log_list_key(session_id: str) -> str:
    return f"session_logs_list:{session_id}"


def get_cancel_key(session_id: str) -> str:
    return f"session_cancel:{session_id}"


def get_pid_key(session_id: str) -> str:
    return f"session_pid:{session_id}"


def push_line(session_id: str, line: str):
    if not session_id:
        return
    line = line.rstrip("\n")
    list_key = get_log_list_key(session_id)
    redis_conn.rpush(list_key, line)
    redis_conn.expire(list_key, _LOG_TTL)
    redis_conn.publish(get_channel_name(session_id), "new")


def finish_session(session_id: str):
    """Appends __FINISHED__ sentinel, notifies listeners, cleans up control keys."""
    list_key = get_log_list_key(session_id)
    redis_conn.rpush(list_key, "__FINISHED__")
    redis_conn.expire(list_key, _LOG_TTL)
    redis_conn.publish(get_channel_name(session_id), "__FINISHED__")
    redis_conn.delete(get_cancel_key(session_id), get_pid_key(session_id))


def cancel_session(session_id: str):
    pid_key = get_pid_key(session_id)
    pid_to_kill = redis_conn.get(pid_key)
    if pid_to_kill:
        _log.info(
            "Cancelling progress session process",
            extra={
                "event": "progress_session.cancel_process",
                "session_id": session_id,
                "pid": pid_to_kill,
            },
        )
        try:
            os.killpg(int(pid_to_kill), signal.SIGTERM)
            _log.info(
                "Sent SIGTERM to progress session process",
                extra={
                    "event": "progress_session.sigterm_sent",
                    "session_id": session_id,
                    "pid": pid_to_kill,
                },
            )
        except (ProcessLookupError, ValueError, PermissionError) as e:
            _log.warning(
                "Could not terminate progress session process",
                extra={
                    "event": "progress_session.sigterm_failed",
                    "session_id": session_id,
                    "pid": pid_to_kill,
                    "exception_type": type(e).__name__,
                },
                exc_info=True,
            )

    redis_conn.set(get_cancel_key(session_id), "1", ex=600)
    push_line(session_id, "[CANCEL] Job cancelled by user. Process terminated.")
    finish_session(session_id)
    return True


def is_cancelled(session_id: str) -> bool:
    return bool(redis_conn.exists(get_cancel_key(session_id)))


def sse_generator(session_id: str, keepalive_secs: int = 15):
    """
    Stream log lines for a session via Server-Sent Events.

    Race-condition-safe design:
      1. Subscribe to the pub/sub channel FIRST so no future signals are missed.
      2. Replay all lines already stored in the Redis list (handles the case
         where the session finished before this connection was established).
      3. Use pub/sub messages purely as wakeup signals; read new lines only from
         the persistent list to guarantee no duplicates and no missed messages.
    """
    channel = get_channel_name(session_id)
    list_key = get_log_list_key(session_id)

    # Step 1 — subscribe before touching the list so no signals are missed.
    pubsub = redis_conn.pubsub()
    pubsub.subscribe(channel)

    _log.info(
        "Subscribed to progress stream",
        extra={"event": "progress_stream.subscribed", "session_id": session_id, "channel": channel},
    )
    yield "data: --- Streaming logs ---\n\n"

    cursor = 0  # next index to read from the Redis list

    while True:
        # Drain all new list items since last read.
        new_items = redis_conn.lrange(list_key, cursor, -1)
        finished = False
        for item in new_items:
            cursor += 1
            if item == "__FINISHED__":
                finished = True
                break
            formatted = "\n".join(f"data: {ln}" for ln in item.split("\n"))
            yield f"{formatted}\n\n"

        if finished:
            _log.info(
                "Progress stream finished",
                extra={
                    "event": "progress_stream.finished",
                    "session_id": session_id,
                    "channel": channel,
                },
            )
            # Named 'done' event tells the client to close the EventSource
            # immediately, preventing auto-reconnect loops.
            yield "event: done\ndata: finished\n\n"
            break

        # Wait for a wakeup signal (new data or __FINISHED__) or send a keepalive.
        msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=keepalive_secs)
        if msg is None:
            yield f": keep-alive at {int(time.time())}\n\n"
        # Whether we got a signal or timed out, loop back and re-read the list.

    pubsub.unsubscribe(channel)
    pubsub.close()

import redis
import time
import os
import signal
from django.conf import settings

redis_conn = redis.from_url(settings.LOGGING_REDIS_URL, decode_responses=True)


def get_channel_name(session_id: str) -> str:
    return f"session_logs:{session_id}"


def get_cancel_key(session_id: str) -> str:
    return f"session_cancel:{session_id}"


def get_pid_key(session_id: str) -> str:
    """Generates the Redis key for storing the process ID."""
    return f"session_pid:{session_id}"


def push_line(session_id: str, line: str):
    if not session_id:
        return
    channel = get_channel_name(session_id)
    line = line.rstrip("\n")
    redis_conn.publish(channel, line)


def finish_session(session_id: str):
    """Notifies listeners and cleans up all session keys."""
    channel = get_channel_name(session_id)
    redis_conn.publish(channel, "__FINISHED__")
    redis_conn.delete(get_cancel_key(session_id), get_pid_key(session_id))


def cancel_session(session_id: str):
    """
    Kills the running process and notifies the user.
    """
    pid_key = get_pid_key(session_id)
    pid_to_kill = redis_conn.get(pid_key)
    if pid_to_kill:
        print(
            f"[cancel_session] Found PID {pid_to_kill} for session {session_id}. Attempting to terminate."
        )
        try:
            os.killpg(int(pid_to_kill), signal.SIGTERM)
            print(f"[cancel_session] Successfully sent SIGTERM to PID {pid_to_kill}.")
        except (ProcessLookupError, ValueError, PermissionError) as e:
            print(f"[cancel_session] Could not kill PID {pid_to_kill}: {e}")

    redis_conn.set(get_cancel_key(session_id), "1", ex=600)

    push_line(session_id, "[CANCEL] Job cancelled by user. Process terminated.")
    finish_session(session_id)
    return True


def is_cancelled(session_id: str) -> bool:
    return bool(redis_conn.exists(get_cancel_key(session_id)))


def sse_generator(session_id: str, keepalive_secs: int = 15):
    """Subscribes to a Redis channel and yields messages for an SSE stream."""
    channel = get_channel_name(session_id)
    pubsub = redis_conn.pubsub()
    pubsub.subscribe(channel)

    print(f"[sse_generator] Subscribed to Redis channel: {channel}")
    yield "data: --- Streaming logs ---\n\n"

    while True:
        message = pubsub.get_message(
            ignore_subscribe_messages=True, timeout=keepalive_secs
        )

        if message:
            data = message["data"]
            if data == "__FINISHED__":
                print(
                    f"[sse_generator] Received __FINISHED__ on {channel}. Closing stream."
                )
                break
            formatted_data = "\n".join([f"data: {line}" for line in data.split("\n")])
            yield f"{formatted_data}\n\n"
        else:
            yield f": keep-alive at {int(time.time())}\n\n"

    print(f"[sse_generator] Unsubscribing from {channel}.")
    pubsub.unsubscribe(channel)
    pubsub.close()

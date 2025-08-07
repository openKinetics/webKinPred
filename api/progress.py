# /home/saleh/webKinPred/api/progress.py (REVISED)
import redis
import time
from django.conf import settings

redis_conn = redis.from_url(settings.LOGGING_REDIS_URL, decode_responses=True)

def get_channel_name(session_id: str) -> str:
    """Generates a consistent channel name for Redis Pub/Sub."""
    return f"session_logs:{session_id}"

def get_cancel_key(session_id: str) -> str:
    """Generates the key used for the cancellation flag."""
    return f"session_cancel:{session_id}"

def push_line(session_id: str, line: str):
    """Publishes a log line to the session's Redis channel."""
    if not session_id:
        return
    channel = get_channel_name(session_id)
    line = line.rstrip("\n")
    # The publish command sends the message to all subscribers
    redis_conn.publish(channel, line)

def finish_session(session_id: str):
    """Notifies listeners that the session is finished by sending a special message."""
    channel = get_channel_name(session_id)
    # This special message will be caught by the generator to close the connection
    redis_conn.publish(channel, "__FINISHED__")
    # Clean up the cancellation key if it exists
    redis_conn.delete(get_cancel_key(session_id))

def cancel_session(session_id: str):
    """
    Sets a cancellation flag in Redis and publishes a final message.
    The running process is expected to check this flag.
    """
    # Set a simple key in Redis. The 'ex' sets an expiration time (e.g., 10 mins)
    # so it gets cleaned up automatically if something goes wrong.
    redis_conn.set(get_cancel_key(session_id), "1", ex=600)
    
    # Push messages to inform the user and to end the stream
    push_line(session_id, "[CANCEL] Job cancelled by user.")
    finish_session(session_id)
    return True

def is_cancelled(session_id: str) -> bool:
    """Checks if the cancellation flag exists in Redis."""
    return bool(redis_conn.exists(get_cancel_key(session_id)))

def sse_generator(session_id: str, keepalive_secs: int = 15):
    """Subscribes to a Redis channel and yields messages for an SSE stream."""
    channel = get_channel_name(session_id)
    pubsub = redis_conn.pubsub()
    pubsub.subscribe(channel)

    print(f"[sse_generator] Subscribed to Redis channel: {channel}")
    yield "data: --- Streaming logs ---\n\n"
    
    while True:
        # The timeout allows the loop to run periodically to send keep-alives
        message = pubsub.get_message(ignore_subscribe_messages=True, timeout=keepalive_secs)
        
        if message:
            data = message['data']
            if data == "__FINISHED__":
                print(f"[sse_generator] Received __FINISHED__ on {channel}. Closing stream.")
                break # Exit the loop to close the stream
            
            # SSE protocol requires special formatting for multi-line data
            formatted_data = "\n".join([f"data: {line}" for line in data.split('\n')])
            yield f"{formatted_data}\n\n"
        else:
            # No message received in 'timeout' seconds, send a comment as a keep-alive
            yield f": keep-alive at {int(time.time())}\n\n"

    print(f"[sse_generator] Unsubscribing from {channel}.")
    pubsub.unsubscribe(channel)
    pubsub.close()

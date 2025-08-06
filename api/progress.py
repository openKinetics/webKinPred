# api/progress.py
import queue
import time
import signal

# session_id -> {"q": Queue, "finished": bool, "last_seen": float, "finished_at": float|None,
#                "procs": set[Popen], "cancelled": bool}
_REGISTRY = {}
_IDLE_TTL_SECS = 600  # 10 minutes

def start_session(session_id: str):
    print(f"[progress.py:start_session] Called for session_id={session_id}")
    cleanup_idle_sessions()
    now = time.time()
    if session_id not in _REGISTRY:
        print(f"[progress.py:start_session] Creating new session entry for {session_id}")
        _REGISTRY[session_id] = {
            "q": queue.Queue(maxsize=1000),
            "finished": False,
            "last_seen": now,
            "finished_at": None,
            "procs": set(),
            "cancelled": False,
        }
    else:
        print(f"[progress.py:start_session] Session {session_id} already exists")

def push_line(session_id: str, line: str):
    print(f"[progress.py:push_line] Attempting to push line for session={session_id}: {line}")
    if session_id not in _REGISTRY:
        print(f"[progress.py:push_line] Session {session_id} not found (ignoring line)")
        return
    _REGISTRY[session_id]["last_seen"] = time.time()
    q = _REGISTRY[session_id]["q"]
    line = line.rstrip("\n")
    try:
        q.put_nowait(line)
        print(f"[progress.py:push_line] Successfully pushed line to session {session_id}")
    except queue.Full:
        print(f"[progress.py:push_line] Queue full for session {session_id}, popping oldest item")
        try:
            q.get_nowait()
        except queue.Empty:
            print(f"[progress.py:push_line] Queue was unexpectedly empty while handling overflow")
        q.put_nowait(line)

def finish_session(session_id: str):
    print(f"[progress.py:finish_session] Called for session_id={session_id}")
    if session_id in _REGISTRY:
        now = time.time()
        _REGISTRY[session_id]["finished"] = True
        _REGISTRY[session_id]["finished_at"] = now
        _REGISTRY[session_id]["last_seen"] = now
        print(f"[progress.py:finish_session] Marked session {session_id} as finished")
    else:
        print(f"[progress.py:finish_session] Session {session_id} not found")
    cleanup_idle_sessions()

def is_finished(session_id: str) -> bool:
    result = session_id in _REGISTRY and _REGISTRY[session_id]["finished"]
    print(f"[progress.py:is_finished] Session {session_id} finished? {result}")
    return result

def is_cancelled(session_id: str) -> bool:
    result = session_id in _REGISTRY and _REGISTRY[session_id]["cancelled"]
    print(f"[progress.py:is_cancelled] Session {session_id} cancelled? {result}")
    return result

def get_queue(session_id: str) -> queue.Queue:
    print(f"[progress.py:get_queue] Called for session_id={session_id}")
    start_session(session_id)
    return _REGISTRY[session_id]["q"]

def register_proc(session_id: str, proc):
    print(f"[progress.py:register_proc] Registering process for session {session_id}")
    if session_id in _REGISTRY:
        _REGISTRY[session_id]["procs"].add(proc)
        print(f"[progress.py:register_proc] Process added to session {session_id}")

def unregister_proc(session_id: str, proc):
    print(f"[progress.py:unregister_proc] Unregistering process for session {session_id}")
    if session_id in _REGISTRY:
        _REGISTRY[session_id]["procs"].discard(proc)
        print(f"[progress.py:unregister_proc] Process removed from session {session_id}")

def cancel_session(session_id: str):
    print(f"[progress.py:cancel_session] Cancelling session {session_id}")
    if session_id not in _REGISTRY:
        print(f"[progress.py:cancel_session] Session {session_id} not found")
        return False
    _REGISTRY[session_id]["cancelled"] = True
    procs = list(_REGISTRY[session_id]["procs"])
    for p in procs:
        try:
            print(f"[progress.py:cancel_session] Terminating process {p.pid} for session {session_id}")
            p.terminate()
        except Exception as e:
            print(f"[progress.py:cancel_session] Exception during terminate: {e}")
    deadline = time.time() + 1.0
    for p in procs:
        try:
            p.wait(timeout=max(0.0, deadline - time.time()))
        except Exception:
            try:
                print(f"[progress.py:cancel_session] Killing process {p.pid} for session {session_id}")
                p.kill()
            except Exception as e:
                print(f"[progress.py:cancel_session] Exception during kill: {e}")
    push_line(session_id, "[CANCEL] Job cancelled by user")
    finish_session(session_id)
    return True

def sse_generator(session_id: str, keepalive_secs: int = 10):
    print(f"[progress.py:sse_generator] Starting SSE for session_id={session_id}")
    q = get_queue(session_id)
    last_emit = time.time()
    yield "data: --- Streaming logs ---\n\n"
    while True:
        try:
            line = q.get(timeout=1.0)
            print(f"[progress.py:sse_generator] Emitting line for {session_id}: {line}")
            last_emit = time.time()
            if session_id in _REGISTRY:
                _REGISTRY[session_id]["last_seen"] = last_emit
            yield f"data: {line}\n\n"
        except queue.Empty:
            now = time.time()
            if now - last_emit >= keepalive_secs:
                last_emit = now
                if session_id in _REGISTRY:
                    _REGISTRY[session_id]["last_seen"] = last_emit
                print(f"[progress.py:sse_generator] Keep-alive event for session {session_id}")
                yield f": keep-alive {int(last_emit)}\n\n"

        if int(time.time()) % 7 == 0:
            print(f"[progress.py:sse_generator] Performing cleanup check for session {session_id}")
            cleanup_idle_sessions()

        if is_finished(session_id) and q.empty():
            print(f"[progress.py:sse_generator] Session {session_id} finished and queue empty. Ending SSE.")
            yield "data: --- End of log ---\n\n"
            break

def cleanup_idle_sessions():
    print("[progress.py:cleanup_idle_sessions] Checking for idle sessions")
    now = time.time()
    doomed = []
    for sid, v in list(_REGISTRY.items()):
        finished = v.get("finished")
        finished_at = v.get("finished_at")
        last_seen = v.get("last_seen", 0)

        if finished and finished_at is not None:
            if now - finished_at > _IDLE_TTL_SECS:
                print(f"[progress.py:cleanup_idle_sessions] Session {sid} idle after finish. Marking for removal.")
                doomed.append(sid)
            continue

        if now - last_seen > _IDLE_TTL_SECS:
            print(f"[progress.py:cleanup_idle_sessions] Session {sid} idle for too long. Marking for removal.")
            doomed.append(sid)

    for sid in doomed:
        print(f"[progress.py:cleanup_idle_sessions] Removing session {sid}")
        _REGISTRY.pop(sid, None)

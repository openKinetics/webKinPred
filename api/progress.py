# api/progress.py
import queue
import time
import signal

# session_id -> {"q": Queue, "finished": bool, "last_seen": float, "finished_at": float|None,
#                "procs": set[Popen], "cancelled": bool}
_REGISTRY = {}
_IDLE_TTL_SECS = 600  # 10 minutes

def start_session(session_id: str):
    cleanup_idle_sessions()
    now = time.time()
    if session_id not in _REGISTRY:
        _REGISTRY[session_id] = {
            "q": queue.Queue(maxsize=1000),
            "finished": False,
            "last_seen": now,
            "finished_at": None,
            "procs": set(),
            "cancelled": False,
        }

def push_line(session_id: str, line: str):
    if session_id not in _REGISTRY:
        return  # don't auto-create on stray pushes
    _REGISTRY[session_id]["last_seen"] = time.time()
    q = _REGISTRY[session_id]["q"]
    line = line.rstrip("\n")
    try:
        q.put_nowait(line)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(line)

def finish_session(session_id: str):
    if session_id in _REGISTRY:
        now = time.time()
        _REGISTRY[session_id]["finished"] = True
        _REGISTRY[session_id]["finished_at"] = now
        _REGISTRY[session_id]["last_seen"] = now
    cleanup_idle_sessions()

def is_finished(session_id: str) -> bool:
    return session_id in _REGISTRY and _REGISTRY[session_id]["finished"]

def is_cancelled(session_id: str) -> bool:
    return session_id in _REGISTRY and _REGISTRY[session_id]["cancelled"]

def get_queue(session_id: str) -> queue.Queue:
    start_session(session_id)
    return _REGISTRY[session_id]["q"]

def register_proc(session_id: str, proc):
    if session_id in _REGISTRY:
        _REGISTRY[session_id]["procs"].add(proc)

def unregister_proc(session_id: str, proc):
    if session_id in _REGISTRY:
        _REGISTRY[session_id]["procs"].discard(proc)

def cancel_session(session_id: str):
    """
    Best-effort: mark cancelled and terminate any running subprocesses.
    """
    if session_id not in _REGISTRY:
        return False
    _REGISTRY[session_id]["cancelled"] = True
    # try graceful terminate; then kill after short delay if still alive
    procs = list(_REGISTRY[session_id]["procs"])
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    # small grace; don't block long
    deadline = time.time() + 1.0
    for p in procs:
        try:
            p.wait(timeout=max(0.0, deadline - time.time()))
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    push_line(session_id, "[CANCEL] Job cancelled by user")
    finish_session(session_id)
    return True

def sse_generator(session_id: str, keepalive_secs: int = 10):
    q = get_queue(session_id)
    last_emit = time.time()
    yield "data: --- Streaming logs ---\n\n"
    while True:
        try:
            line = q.get(timeout=1.0)
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
                yield f": keep-alive {int(last_emit)}\n\n"

        # opportunistic clean-up
        if int(time.time()) % 7 == 0:
            cleanup_idle_sessions()

        if is_finished(session_id) and q.empty():
            yield "data: --- End of log ---\n\n"
            break

def cleanup_idle_sessions():
    now = time.time()
    doomed = []
    for sid, v in list(_REGISTRY.items()):
        finished = v.get("finished")
        finished_at = v.get("finished_at")
        last_seen = v.get("last_seen", 0)

        if finished and finished_at is not None:
            if now - finished_at > _IDLE_TTL_SECS:
                doomed.append(sid)
            continue

        if now - last_seen > _IDLE_TTL_SECS:
            doomed.append(sid)

    for sid in doomed:
        _REGISTRY.pop(sid, None)

import subprocess
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import SEQMAP_CLI, SEQMAP_DB, SEQMAP_PY

SEQMAP_CLI = (
    os.environ.get("KINFORM_TOOLS_PATH", "/home/saleh/webKinPred/tools")
    + "/seqmap/main.py"
)
SEQMAP_DB = (
    os.environ.get("KINFORM_MEDIA_PATH", "/home/saleh/webKinPred/media")
    + "/sequence_info/seqmap.sqlite3"
)
SEQMAP_PY = sys.executable

def resolve_seq_ids_via_cli(sequences):
    """Resolve IDs for all sequences in order (increments uses_count per occurrence)."""
    payload = "\n".join(sequences) + "\n"
    cmd = [SEQMAP_PY, SEQMAP_CLI, "--db", SEQMAP_DB, "batch-get-or-create", "--stdin"]
    proc = subprocess.run(
        cmd, input=payload, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"seqmap CLI failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    ids = proc.stdout.strip().splitlines()
    if len(ids) != len(sequences):
        raise RuntimeError(
            f"seqmap returned {len(ids)} ids for {len(sequences)} sequences"
        )
    return ids

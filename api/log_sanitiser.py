# api/log_sanitiser.py
import os
import re
from typing import Dict

# Precompile general path patterns
RE_CONDA = re.compile(r"/home/[^/\s]+/anaconda3/bin/conda")
RE_APP_ROOT = re.compile(r"/home/[^/\s]+/webKinPred")
RE_HOME = re.compile(r"/home/[^/\s]+")
RE_TMP_PATHS = [
    re.compile(r"/tmp/[^\s]+"),
    re.compile(r"/home/saleh/mmseqs_tmp[^\s]*"),
]

# Fallback: any other absolute path (be conservative)
RE_ABS_FALLBACK = re.compile(r"(?<![\w.-])/(?:[^\s]+)")

COMMON_LABELS = [
    (re.compile(r"\bqueryDB_h\b"), "[QUERY_DB_H]"),
    (re.compile(r"\bresultDB_h\b"), "[RESULT_DB_H]"),
    (re.compile(r"\bqueryDB\b"), "[QUERY_DB]"),
    (re.compile(r"\bresultDB\b"), "[RESULT_DB]"),
    (re.compile(r"\bresult\.m8\b"), "[RESULT_M8]"),
    (re.compile(r"\bpref_\d+\b"), "[PREF_DB]"),  
]

def _normalise_target_db_refs(line: str, target_dbs: Dict[str, str] | None) -> str:
    """
    Replace any known target DB paths (and their basenames) with friendly labels:
      /.../targetdb_dlkcat  -> TARGET_DB(DLKcat)  (or whatever key name you use)
    """
    if not target_dbs:
        return line

    for method, path in target_dbs.items():
        label = f"TARGET_DB({method})"
        # Full path replacement
        if path:
            line = line.replace(path, label)
        # Also replace by basename if it shows up
        base = os.path.basename(path.rstrip("/")) if path else ""
        if base:
            line = re.sub(rf"\b{re.escape(base)}\b", label, line)
    return line


def sanitise_log_line(line: str, target_dbs: Dict[str, str] | None = None) -> str:
    """
    Apply the requested redactions, while keeping the log readable and useful.
    Order matters: normalise known DBs first, then scrub paths.
    """
    # 1) Normalise target DBs to friendly labels
    line = _normalise_target_db_refs(line, target_dbs)
    # 2) Temporary paths: replace with [TMP] 
    for tmp_re in RE_TMP_PATHS:
        line = tmp_re.sub("[TMP]", line)
    # 3) General path patterns
    line = RE_CONDA.sub("[CONDA]", line)          # conda binary path
    line = RE_APP_ROOT.sub("[APP_ROOT]", line)    # project root
    line = RE_HOME.sub("[HOME]", line)            # any /home/<user>
    # 4) Any other absolute paths that slipped through
    line = RE_ABS_FALLBACK.sub("[PATH]", line)
    # 5) Common ephemeral artefact names -> friendly labels
    for regex, replacement in COMMON_LABELS:
        line = regex.sub(replacement, line)

    return line

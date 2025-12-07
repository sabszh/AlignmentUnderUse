"""
Helpers for writing JSONL data to disk and managing output folders.
"""

import json
import os
from typing import Any, Dict


def ensure_dir(path: str) -> None:
    """
    Ensure a directory exists (create it if missing).

    Args:
        path (str): Directory path to create if it does not already exist.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def write_jsonl(path_str: str, obj: Dict[str, Any]) -> None:
    """
    Append a single JSON object as one line to a JSONL file.

    Args:
        path_str (str): Path to the JSONL file.
        obj (dict): JSON-serializable Python dict to write.
    """
    with open(path_str, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

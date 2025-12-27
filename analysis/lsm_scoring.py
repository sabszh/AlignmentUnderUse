"""
Linguistic Style Matching (LSM) computation script.

Loads conversations and computes LSM (Linguistic Style Matching) scores between
sequential user-assistant message pairs. Outputs a simple dataset with:
- conv_id: conversation identifier
- turn: sequential turn number (pair_step) within conversation
- lsm_score: overall LSM score for that turn pair

LSM measures linguistic style similarity across functional word categories:
articles, prepositions, pronouns, aux_verbs, conjunctions, negations, adverbs.

Usage:
    python -m analysis.lsm_scoring
    python -m analysis.lsm_scoring --input ../data/processed/conversations_english.jsonl --output ../data/outputs/lsm_scores.csv
"""

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# --- Configuration ---

FUNCTIONAL_WORDS = {
    "articles": {"a", "an", "the"},
    "prepositions": {
        "about", "above", "across", "after", "against", "along", "among", "around", "at", "before",
        "behind", "below", "beneath", "beside", "between", "beyond", "but", "by", "concerning",
        "despite", "down", "during", "except", "for", "from", "in", "inside", "into", "like",
        "near", "of", "off", "on", "onto", "out", "outside", "over", "past", "regarding",
        "since", "through", "throughout", "to", "toward", "under", "underneath", "until", "up",
        "upon", "with", "within", "without"
    },
    "pronouns": {
        "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "this", "that", "these", "those"
    },
    "aux_verbs": {
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having",
        "do", "does", "did", "doing",
        "will", "would", "shall", "should", "can", "could", "may", "might", "must"
    },
    "conjunctions": {
        "and", "but", "or", "nor", "for", "so", "yet", "although", "because",
        "since", "unless", "while", "whereas", "if", "though"
    },
    "negations": {
        "no", "not", "never", "none", "nobody", "nothing", "neither", "nowhere",
        "cannot", "can't", "won't", "don't", "doesn't", "didn't", "isn't", "aren't",
        "wasn't", "weren't", "haven't", "hasn't", "hadn't", "shouldn't", "wouldn't",
        "couldn't", "mustn't"
    },
    "adverbs_common": {
        "very", "really", "just", "quite", "rather", "too", "also", "still", "even",
        "only", "almost", "already", "soon", "then", "there", "here", "often", "always",
        "sometimes", "usually", "maybe", "perhaps"
    },
}

FW_KEYS = list(FUNCTIONAL_WORDS.keys())
WORD_RE = re.compile(r"[a-zA-Z']+")


# --- Parsing helpers ---

def parse_messages(x: Any) -> List[Dict]:
    """Parse messages column robustly."""
    if isinstance(x, list):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if not isinstance(x, str):
        return []
    s = x.strip()
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def count_turns(messages: List[Dict]) -> int:
    """Count user+assistant turns with non-empty text."""
    if not isinstance(messages, list):
        return 0
    n = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        text = m.get("text")
        if role in {"user", "assistant"} and isinstance(text, str) and text.strip():
            n += 1
    return n


# --- LSM computation ---

def tokenize_words(text: str) -> List[str]:
    """Tokenize text to lowercase words."""
    if not isinstance(text, str) or not text.strip():
        return []
    return [w.lower() for w in WORD_RE.findall(text)]


def fw_proportions(text: str) -> Dict[str, float]:
    """Compute functional word proportions for a text."""
    toks = tokenize_words(text)
    n = len(toks)
    if n == 0:
        return {k: 0.0 for k in FW_KEYS} | {"n_tokens": 0}
    
    counts = {k: 0 for k in FW_KEYS}
    for w in toks:
        for k in FW_KEYS:
            if w in FUNCTIONAL_WORDS[k]:
                counts[k] += 1
    
    props = {k: counts[k] / n for k in FW_KEYS}
    props["n_tokens"] = n
    return props


def lsm_score(p1: float, p2: float, eps: float = 1e-8) -> float:
    """Compute LSM score between two proportions."""
    return 1.0 - (abs(p1 - p2) / (p1 + p2 + eps))


def explode_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Explode messages to one row per message."""
    conv_id_col = None
    for c in ["conversation_id", "share_id", "conv_id", "id", "url"]:
        if c in df.columns:
            conv_id_col = c
            break
    
    rows = []
    for conv_idx, row in df.iterrows():
        conv_id = row[conv_id_col] if conv_id_col else conv_idx
        msgs = row.get("messages", [])
        if not isinstance(msgs, list):
            continue
        
        for mi, msg in enumerate(msgs):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            text = msg.get("text")
            if role in {"user", "assistant"} and isinstance(text, str) and text.strip():
                rows.append({
                    "conv_id": conv_id,
                    "msg_idx": mi,
                    "role": role,
                    "text": text
                })
    
    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows).sort_values(["conv_id", "msg_idx"]).reset_index(drop=True)


def compute_lsm(df_conv: pd.DataFrame) -> pd.DataFrame:
    """Compute LSM scores for all turn pairs in conversations."""
    df_msgs = explode_messages(df_conv)
    
    if df_msgs.empty:
        return pd.DataFrame(columns=["conv_id", "turn", "lsm_score"])
    
    # Compute functional word proportions
    fw_df = df_msgs["text"].apply(fw_proportions).apply(pd.Series)
    df_msgs_fw = pd.concat([df_msgs, fw_df], axis=1)
    
    # Compute LSM pairs
    pair_rows = []
    for conv_id, g in df_msgs_fw.groupby("conv_id", sort=False):
        g = g.sort_values("msg_idx").reset_index(drop=True)
        turn = 0
        
        for i in range(1, len(g)):
            r_prev, r_now = g.loc[i - 1, "role"], g.loc[i, "role"]
            
            # Only count transitions between different roles
            if r_prev == r_now:
                continue
            
            turn += 1
            
            # Compute LSM for this pair
            cat_scores = []
            for k in FW_KEYS:
                s = lsm_score(g.loc[i - 1, k], g.loc[i, k])
                cat_scores.append(s)
            
            lsm = float(np.mean(cat_scores))
            
            pair_rows.append({
                "conv_id": conv_id,
                "turn": turn,
                "lsm_score": lsm,
            })
    
    if not pair_rows:
        return pd.DataFrame(columns=["conv_id", "turn", "lsm_score"])
    
    return pd.DataFrame(pair_rows)


# --- Main ---

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute Linguistic Style Matching (LSM) scores for conversations"
    )
    
    parser.add_argument(
        "--input",
        default="../data/processed/conversations_english.jsonl",
        help="Input JSONL with conversations (default: ../data/processed/conversations_english.jsonl)",
    )
    
    parser.add_argument(
        "--output",
        default="../data/outputs/lsm_scores.csv",
        help="Output CSV with LSM scores (default: ../data/outputs/lsm_scores.csv)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / args.input
    output_path = base_dir / args.output
    
    if not input_path.exists():
        # Try legacy location
        legacy_input = base_dir / "../data/conversations_english.jsonl"
        if legacy_input.exists():
            input_path = legacy_input
        else:
            raise FileNotFoundError(f"Input not found: {input_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[lsm_scoring] Loading conversations from: {input_path}")
    df_conv = pd.read_json(str(input_path), lines=True)
    
    print(f"[lsm_scoring] Loaded {len(df_conv)} conversations")
    
    # Parse messages
    df_conv["messages"] = df_conv["messages"].apply(parse_messages)
    df_conv["n_turns"] = df_conv["messages"].apply(count_turns)
    
    print(f"[lsm_scoring] Turn statistics:")
    print(df_conv["n_turns"].describe())
    
    # Compute LSM
    print(f"[lsm_scoring] Computing LSM scores...")
    df_lsm = compute_lsm(df_conv)
    
    # Save
    df_lsm.to_csv(output_path, index=False)
    print(f"[lsm_scoring] Saved {len(df_lsm)} LSM pairs to: {output_path}")
    
    # Summary
    print(f"\n[lsm_scoring] LSM Summary:")
    print(df_lsm["lsm_score"].describe())


if __name__ == "__main__":
    main()

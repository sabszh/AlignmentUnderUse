"""
Lexical and Syntactic Alignment computation for ChatGPT conversation turn pairs.

This script computes:
- Lexical alignment: Jaccard similarity between user and assistant message word sets.
- Syntactic alignment: Jaccard similarity between user and assistant message POS tag sets (using spaCy).

Outputs a CSV with turn pairs and alignment scores.

Usage:
    python -m src.measures.lexsyn_alignment --input data/processed/conversations_english.jsonl --output data/derived/lexsyn_alignment.csv

Notes:
- Requires spaCy and the en_core_web_sm model: pip install spacy && python -m spacy download en_core_web_sm
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import spacy
from tqdm.auto import tqdm

from ..schemas.turn import TURN_SCHEMA

import os
# Use GPU if available and requested
if spacy.prefer_gpu():
    print("[lexsyn_alignment] Using spaCy GPU mode.")
else:
    print("[lexsyn_alignment] Using spaCy CPU mode.")
nlp = spacy.load("en_core_web_sm")

# --- Parsing helpers ---
def parse_messages(x: Any) -> List[Dict]:
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
        return []

# --- Alignment functions ---
def lexical_jaccard(u_text: str, a_text: str) -> float:
    u_words = set(str(u_text).lower().split())
    a_words = set(str(a_text).lower().split())
    if not u_words and not a_words:
        return np.nan
    return len(u_words & a_words) / len(u_words | a_words)


# Batch POS tagging for speed
def batch_pos_jaccard(user_texts, assistant_texts):
    # Use spaCy's nlp.pipe for batch processing
    user_docs = list(tqdm(nlp.pipe(user_texts, batch_size=128, disable=["ner"]), total=len(user_texts), desc="spaCy user POS"))
    assistant_docs = list(tqdm(nlp.pipe(assistant_texts, batch_size=128, disable=["ner"]), total=len(assistant_texts), desc="spaCy assistant POS"))
    pos_jaccards = []
    for idx, (u_doc, a_doc) in enumerate(tqdm(zip(user_docs, assistant_docs), total=len(user_docs), desc="POS Jaccard")):
        u_pos = set([token.pos_ for token in u_doc])
        a_pos = set([token.pos_ for token in a_doc])
        if not u_pos and not a_pos:
            pos_jaccards.append(np.nan)
        else:
            pos_jaccards.append(len(u_pos & a_pos) / len(u_pos | a_pos))
    return pos_jaccards

# --- Explode messages ---
def explode_messages(df: pd.DataFrame) -> pd.DataFrame:
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
                    "text": text,
                    "message_id": f"{conv_id}_{mi}_{role}",
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["conv_id", "msg_idx"]).reset_index(drop=True)

# --- Turn pair creation ---
def create_turn_pairs(df_msgs: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    for conv_id, g in df_msgs.groupby("conv_id", sort=False):
        g = g.sort_values("msg_idx").reset_index(drop=True)
        user_turn = 0
        asst_turn = 0
        for i in range(1, len(g)):
            r_prev, r_now = g.loc[i - 1, "role"], g.loc[i, "role"]
            if r_prev == r_now:
                continue
            if r_prev == "user" and r_now == "assistant":
                user_turn += 1
                turn_idx = user_turn
                direction = "user→assistant"
                user_row = g.loc[i - 1]
                assistant_row = g.loc[i]
            elif r_prev == "assistant" and r_now == "user":
                asst_turn += 1
                turn_idx = -asst_turn
                direction = "assistant→user"
                user_row = g.loc[i]
                assistant_row = g.loc[i - 1]
            else:
                continue
            pairs.append({
                "conv_id": conv_id,
                "turn": turn_idx,
                "direction": direction,
                "user_message_id": user_row["message_id"],
                "assistant_message_id": assistant_row["message_id"],
                "user_text": user_row["text"],
                "assistant_text": assistant_row["text"],
            })
    if not pairs:
        return pd.DataFrame()
    return pd.DataFrame(pairs)

def compute_alignment(df_pairs: pd.DataFrame) -> pd.DataFrame:
# --- Alignment computation ---
    print("[lexsyn_alignment] Computing lexical and syntactic alignment...")
    df_pairs = df_pairs.copy()
    # Lexical Jaccard with tqdm
    # Lexical Jaccard with tqdm and periodic saving
    lexical_scores = []
    save_every = 5000
    for idx, (u, a) in enumerate(tqdm(zip(df_pairs["user_text"], df_pairs["assistant_text"]), total=len(df_pairs), desc="Lexical Jaccard")):
        lexical_scores.append(lexical_jaccard(u, a))
        if (idx + 1) % save_every == 0:
            df_pairs.loc[:idx, "lexical_jaccard"] = lexical_scores
            partial = df_pairs.loc[:idx, ["conv_id", "turn", "direction", "user_message_id", "assistant_message_id", "lexical_jaccard"]]
            partial.to_csv("data/derived/lexsyn_alignment_partial.csv", index=False)
    df_pairs["lexical_jaccard"] = lexical_scores
    partial = df_pairs[["conv_id", "turn", "direction", "user_message_id", "assistant_message_id", "lexical_jaccard"]]
    partial.to_csv("data/derived/lexsyn_alignment_partial.csv", index=False)

    # Syntactic Jaccard (POS) with tqdm and periodic saving
    pos_scores = []
    user_texts = list(df_pairs["user_text"])
    assistant_texts = list(df_pairs["assistant_text"])
    user_docs = list(tqdm(nlp.pipe(user_texts, batch_size=128, disable=["ner"]), total=len(user_texts), desc="spaCy user POS"))
    assistant_docs = list(tqdm(nlp.pipe(assistant_texts, batch_size=128, disable=["ner"]), total=len(assistant_texts), desc="spaCy assistant POS"))
    for idx, (u_doc, a_doc) in enumerate(tqdm(zip(user_docs, assistant_docs), total=len(user_docs), desc="POS Jaccard")):
        u_pos = set([token.pos_ for token in u_doc])
        a_pos = set([token.pos_ for token in a_doc])
        if not u_pos and not a_pos:
            pos_scores.append(np.nan)
        else:
            pos_scores.append(len(u_pos & a_pos) / len(u_pos | a_pos))
        if (idx + 1) % save_every == 0:
            df_pairs.loc[:idx, "pos_jaccard"] = pos_scores
            partial = df_pairs.loc[:idx, ["conv_id", "turn", "direction", "user_message_id", "assistant_message_id", "lexical_jaccard", "pos_jaccard"]]
            partial.to_csv("data/derived/lexsyn_alignment_partial.csv", index=False)
    df_pairs["pos_jaccard"] = pos_scores
    partial = df_pairs[["conv_id", "turn", "direction", "user_message_id", "assistant_message_id", "lexical_jaccard", "pos_jaccard"]]
    partial.to_csv("data/derived/lexsyn_alignment_partial.csv", index=False)
    print("  ✓ Computed alignment for", len(df_pairs), "pairs")
    return df_pairs

# --- Main ---
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Lexical and Syntactic Alignment for conversation turn pairs"
    )
    parser.add_argument(
        "--input",
        default="data/processed/conversations_english.jsonl",
        help="Input JSONL with conversations (default: data/processed/conversations_english.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="data/derived/lexsyn_alignment.csv",
        help="Output CSV with alignment scores (default: data/derived/lexsyn_alignment.csv)",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[lexsyn_alignment] Loading conversations from: {input_path}")
    df_conv = pd.read_json(str(input_path), lines=True)
    df_conv["messages"] = df_conv["messages"].apply(parse_messages)
    df_msgs = explode_messages(df_conv)
    print(f"[lexsyn_alignment] Loaded {len(df_msgs)} messages")
    df_pairs = create_turn_pairs(df_msgs)
    print(f"[lexsyn_alignment] Created {len(df_pairs)} turn pairs")
    df_pairs = compute_alignment(df_pairs)
    output_cols = TURN_SCHEMA + ["lexical_jaccard", "pos_jaccard"]
    df_pairs = df_pairs[[c for c in output_cols if c in df_pairs.columns]]
    df_pairs.to_csv(output_path, index=False)
    print(f"[lexsyn_alignment] Saved {len(df_pairs)} pairs to: {output_path}")
    print("\n[lexsyn_alignment] Alignment Summary:")
    print(df_pairs[["lexical_jaccard", "pos_jaccard"]].describe())

if __name__ == "__main__":
    main()

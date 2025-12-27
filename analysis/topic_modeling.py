"""
Topic modeling for ChatGPT conversations using Turftopic KeyNMF.

Builds three independent topic models over different views of each conversation:
- User-only text (what users ask about)
- Assistant-only text (how the assistant responds)
- Combined text (full conversation: user + assistant)

Outputs a CSV with one dominant topic assignment per model for every conversation,
including the topic ID, keywords, and confidence score. Optionally saves a plot
of the top topics per model.

Usage:
    python -m analysis.topic_modeling
    python -m analysis.topic_modeling --input ../data/conversations_english.jsonl --topics 30 --keywords 9
    python -m analysis.topic_modeling --plot --output-dir topic_model_output

Notes:
- Turftopic's KeyNMF returns topics as a list of (topic_id, [(word, score), ...]).
  We convert this to a dict for convenient keyword lookup.
- Topic IDs are independent across the three models; same numeric ID does not
  imply the same concept between user/assistant/combined models.
"""

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from turftopic import KeyNMF
except Exception as e:
    KeyNMF = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


# ---------------------------
# Parsing helpers
# ---------------------------

def parse_maybe_json(x: Any) -> Optional[Any]:
    """Parse x which might be JSON string, python-literal string, or already a list.

    Returns native Python object (list/dict) or None.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (list, dict)):
        return x
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def extract_text_content(text_or_obj: Any) -> str:
    """Extract plain text from a message text field.

    Skips metadata/asset pointers and serialized dicts; returns only user-visible text.
    """
    if not text_or_obj:
        return ""
    if isinstance(text_or_obj, dict):
        return ""
    if not isinstance(text_or_obj, str):
        return ""
    s = text_or_obj.strip()
    if s.startswith("{") or s.startswith("["):
        return ""
    return s


def build_conversation_doc(messages: Any, mode: str = "user_only") -> str:
    """Create one document per conversation for KeyNMF.

    mode: 'user_only', 'assistant_only', or 'all_turns'
    """
    if not isinstance(messages, list):
        return ""
    parts: List[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        txt = (m.get("text", "") or "")
        txt = extract_text_content(txt)
        if not txt:
            continue
        if mode == "user_only" and role != "user":
            continue
        if mode == "assistant_only" and role != "assistant":
            continue
        if mode == "all_turns" and role not in ("user", "assistant"):
            continue
        parts.append(txt)
    return "\n".join(parts).strip()


# ---------------------------
# Topic helpers
# ---------------------------

def get_topic_keywords(model: KeyNMF, tid: int, n: int) -> List[str]:
    """Extract top n keywords for a topic from KeyNMF model."""
    try:
        topics = model.get_topics()  # list of (topic_id, [(word, score), ...])
        topics_dict = {topic_id: word_scores for topic_id, word_scores in topics}
        items = topics_dict.get(tid, [])
        if not items:
            return []
        if isinstance(items[0], (tuple, list)):
            return [str(w) for w, *_ in items[:n]]
        return list(items[:n])
    except Exception:
        return []


def build_topic_keyword_map(model: KeyNMF, n_topics: int, n_keywords: int) -> Dict[int, str]:
    """Build mapping of topic IDs to a display string of keywords."""
    topic_kw_map: Dict[int, str] = {}
    for tid in range(n_topics):
        keywords = get_topic_keywords(model, tid, n=n_keywords)
        topic_kw_map[tid] = " / ".join(keywords) if keywords else f"Topic_{tid}"
    return topic_kw_map


def assign_topics(doc_topic_matrix: np.ndarray, model: KeyNMF, prefix: str, n_topics: int, n_keywords: int) -> pd.DataFrame:
    """Assign dominant topic + keywords for each conversation for a given model."""
    topic_ids = doc_topic_matrix.argmax(axis=1).astype(int)
    topic_scores = doc_topic_matrix.max(axis=1).astype(float)
    topic_kw_map = build_topic_keyword_map(model, n_topics=n_topics, n_keywords=n_keywords)
    keywords_list = [topic_kw_map.get(int(tid), f"Topic_{int(tid)}") for tid in topic_ids]
    return pd.DataFrame({
        f"{prefix}_topic_id": topic_ids,
        f"{prefix}_topic_score": topic_scores,
        f"{prefix}_keywords": keywords_list,
    })


# ---------------------------
# Core pipeline
# ---------------------------

def run_pipeline(input_path: Path, output_dir: Path, n_topics: int, n_keywords: int, max_chars_per_doc: int, make_plot: bool) -> Tuple[Path, Optional[Path]]:
    """Run the 3-model topic pipeline and write outputs.

    Returns: (csv_path, plot_path or None)
    """
    if KeyNMF is None:
        raise RuntimeError(f"turftopic import failed: {_IMPORT_ERR}")

    # Load conversations
    df_english = pd.read_json(str(input_path), lines=True)
    if "conv_id" not in df_english.columns:
        if "share_id" in df_english.columns:
            df_english["conv_id"] = df_english["share_id"]
        else:
            raise ValueError("Input must have 'conv_id' or 'share_id'.")
    if "messages" not in df_english.columns:
        raise ValueError("Input must contain a 'messages' column.")

    # Ensure parsed messages
    if isinstance(df_english.loc[df_english.index[0], "messages"], str):
        df_english["messages_parsed"] = df_english["messages"].apply(parse_maybe_json)
    else:
        df_english["messages_parsed"] = df_english["messages"]

    parse_rate = df_english["messages_parsed"].apply(lambda x: isinstance(x, list)).mean()
    print(f"[topic_modeling] Parsed messages success rate: {parse_rate:.3f}")

    # Build documents
    df_docs = df_english[["conv_id", "messages_parsed"]].copy()
    df_docs["doc_user"] = df_docs["messages_parsed"].apply(lambda m: build_conversation_doc(m, mode="user_only"))
    df_docs["doc_user"] = df_docs["doc_user"].fillna("").astype(str).str.strip().str.slice(0, max_chars_per_doc)
    df_docs["doc_assistant"] = df_docs["messages_parsed"].apply(lambda m: build_conversation_doc(m, mode="assistant_only"))
    df_docs["doc_assistant"] = df_docs["doc_assistant"].fillna("").astype(str).str.strip().str.slice(0, max_chars_per_doc)
    df_docs["doc_combined"] = df_docs["messages_parsed"].apply(lambda m: build_conversation_doc(m, mode="all_turns"))
    df_docs["doc_combined"] = df_docs["doc_combined"].fillna("").astype(str).str.strip().str.slice(0, max_chars_per_doc)

    # Keep only conversations with text in all three
    df_docs = df_docs[(df_docs["doc_user"].str.len() > 0) & (df_docs["doc_assistant"].str.len() > 0) & (df_docs["doc_combined"].str.len() > 0)].reset_index(drop=True)
    print(f"[topic_modeling] Valid conversations: {len(df_docs)}")

    # Fit models
    corpus_user = df_docs["doc_user"].tolist()
    corpus_assistant = df_docs["doc_assistant"].tolist()
    corpus_combined = df_docs["doc_combined"].tolist()

    print("[topic_modeling] Fitting user-only KeyNMF...")
    model_user = KeyNMF(n_topics)
    doc_topic_user = model_user.fit_transform(corpus_user)

    print("[topic_modeling] Fitting assistant-only KeyNMF...")
    model_assistant = KeyNMF(n_topics)
    doc_topic_assistant = model_assistant.fit_transform(corpus_assistant)

    print("[topic_modeling] Fitting combined KeyNMF...")
    model_combined = KeyNMF(n_topics)
    doc_topic_combined = model_combined.fit_transform(corpus_combined)

    # Assign topics
    df_user_topics = assign_topics(np.asarray(doc_topic_user), model_user, "user", n_topics, n_keywords)
    df_assistant_topics = assign_topics(np.asarray(doc_topic_assistant), model_assistant, "assistant", n_topics, n_keywords)
    df_combined_topics = assign_topics(np.asarray(doc_topic_combined), model_combined, "combined", n_topics, n_keywords)

    # Merge and save
    df_final = df_docs[["conv_id"]].copy()
    df_final = pd.concat([df_final, df_user_topics, df_assistant_topics, df_combined_topics], axis=1)
    csv_dir = output_dir
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "conversations_with_topics.csv"
    df_final.to_csv(csv_path, index=False)
    print(f"[topic_modeling] Saved: {csv_path}")

    # Optional plot
    plot_path: Optional[Path] = None
    if make_plot:
        if plt is None:
            print("[topic_modeling] matplotlib not available; skipping plot.")
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for idx, (prefix, name) in enumerate([("user", "User Topics"), ("assistant", "Assistant Topics"), ("combined", "Combined Topics")]):
                ax = axes[idx]
                topic_counts = df_final[f"{prefix}_topic_id"].value_counts().head(10).sort_values()
                topic_ids = topic_counts.index
                counts = topic_counts.values
                keywords_list = []
                for tid in topic_ids:
                    kw = df_final[df_final[f"{prefix}_topic_id"] == tid][f"{prefix}_keywords"].iloc[0]
                    keywords_list.append(kw)
                colors = plt.cm.viridis(np.linspace(0, 1, len(topic_ids)))
                bars = ax.barh(range(len(topic_ids)), counts, color=colors)
                labels = [f"T{tid}: {kw}" for tid, kw in zip(topic_ids, keywords_list)]
                ax.set_yticks(range(len(topic_ids)))
                ax.set_yticklabels(labels, fontsize=9)
                ax.set_xlabel("Number of Conversations", fontsize=11)
                ax.set_title(name, fontsize=12, fontweight="bold")
                ax.grid(axis="x", alpha=0.3)
                for i, (bar, count) in enumerate(zip(bars, counts)):
                    ax.text(count, i, f" {count}", va="center", fontsize=9)
            plt.tight_layout()
            plot_path = csv_dir / "topic_distributions.png"
            plt.savefig(str(plot_path), dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[topic_modeling] Saved plot: {plot_path}")

    return csv_path, plot_path


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Topic modeling for ChatGPT conversations using Turftopic KeyNMF"
    )
    parser.add_argument(
        "--input",
        default="../data/processed/conversations_english.jsonl",
        help="Path to input conversations JSONL (default: ../data/processed/conversations_english.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        default="../data/outputs/topics",
        help="Directory to write outputs (CSV/plot). Default: ../data/outputs/topics",
    )
    parser.add_argument(
        "--topics",
        type=int,
        default=30,
        help="Number of topics per model (default: 30)",
    )
    parser.add_argument(
        "--keywords",
        type=int,
        default=9,
        help="Number of keywords per topic (default: 9)",
    )
    parser.add_argument(
        "--max-chars-per-doc",
        type=int,
        default=20000,
        help="Max characters per document (default: 20000)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save topic distribution plot as PNG",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / args.input
    # Fallback to legacy location if missing
    if not input_path.exists():
        legacy_input = base_dir / "../data/conversations_english.jsonl"
        if legacy_input.exists():
            print(f"[topic_modeling] Input not found; using legacy path: {legacy_input}")
            input_path = legacy_input

    output_dir = base_dir / args.output_dir

    if KeyNMF is None:
        raise RuntimeError(
            f"turftopic not installed or failed to import: {_IMPORT_ERR}.\n"
            "Install with: pip install turftopic"
        )

    csv_path, plot_path = run_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        n_topics=args.topics,
        n_keywords=args.keywords,
        max_chars_per_doc=args.max_chars_per_doc,
        make_plot=bool(args.plot),
    )

    print("\n[topic_modeling] Done.")
    print(f"CSV: {csv_path}")
    if plot_path:
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()

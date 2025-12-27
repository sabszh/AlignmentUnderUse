"""
Sentiment alignment computation for ChatGPT conversation turn pairs.

This script computes sentiment similarity between user and assistant messages using
a pretrained sentiment analysis model (distilbert-base-uncased-finetuned-sst-2-english).
It analyzes sentiment polarity for each message and computes similarity scores.

Processing steps:
1. Loads turn pairs from semantic alignment output (or creates them from scratch)
2. Extracts unique messages and computes sentiment scores
3. Maps sentiment to polarity in [-1, 1] range (negative to positive)
4. Computes sentiment similarity for each turn pair as 1 - |difference|/2
5. Saves turn-level data with sentiment similarity scores

Sentiment similarity interpretation:
- 1.0: Identical sentiment (both very positive or both very negative)
- 0.5: Moderate difference (e.g., neutral vs somewhat positive)
- 0.0: Opposite sentiments (very negative vs very positive)

Usage:
    python -m analysis.sentiment_alignment
    python -m analysis.sentiment_alignment --input analysis/semantic_alignment.csv
    python -m analysis.sentiment_alignment --output analysis/sentiment_alignment.csv
    python -m analysis.sentiment_alignment --from-conversations data/conversations_english.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import pipeline
from tqdm.auto import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute sentiment alignment for conversation turn pairs"
    )
    
    parser.add_argument(
        "--input",
        default="analysis/semantic_alignment.csv",
        help="Input CSV with turn pairs from semantic_alignment.py (default: analysis/semantic_alignment.csv)",
    )
    
    parser.add_argument(
        "--from-conversations",
        default=None,
        help="Alternatively, load from conversations JSONL and create pairs (skips semantic similarity)",
    )
    
    parser.add_argument(
        "--output",
        default="analysis/sentiment_alignment.csv",
        help="Output CSV file for turn-level sentiment alignment (default: analysis/sentiment_alignment.csv)",
    )
    
    parser.add_argument(
        "--cache-dir",
        default="data",
        help="Directory for caching sentiment scores (default: data)",
    )
    
    parser.add_argument(
        "--model",
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Sentiment analysis model (default: distilbert-base-uncased-finetuned-sst-2-english)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for sentiment computation (default: 64)",
    )
    
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of sentiment (ignore cache)",
    )
    
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for computation (default: auto)",
    )
    
    return parser.parse_args()


def load_turn_pairs(input_path):
    """Load turn pairs from CSV.
    
    Args:
        input_path: Path to semantic alignment CSV
        
    Returns:
        DataFrame with turn pairs
    """
    print(f"[sentiment_alignment] Loading turn pairs from: {input_path}")
    df_pairs = pd.read_csv(input_path)
    print(f"  Loaded {len(df_pairs):,} turn pairs")
    return df_pairs


def load_from_conversations(conversations_path):
    """Load conversations and create turn pairs (without semantic similarity).
    
    Args:
        conversations_path: Path to conversations JSONL
        
    Returns:
        DataFrame with turn pairs (no semantic_similarity column)
    """
    print(f"[sentiment_alignment] Loading conversations from: {conversations_path}")
    
    # Import the functions from semantic_alignment
    from analysis.semantic_alignment import load_messages, create_turn_pairs
    
    df_messages = load_messages(conversations_path)
    df_pairs = create_turn_pairs(df_messages)
    
    return df_pairs, df_messages


def extract_unique_messages(df_pairs):
    """Extract unique messages from turn pairs.
    
    Args:
        df_pairs: DataFrame with turn pairs
        
    Returns:
        DataFrame with unique messages (message_id, text)
    """
    # Combine user and assistant messages
    user_msgs = df_pairs[['user_message_id', 'user_text']].rename(
        columns={'user_message_id': 'message_id', 'user_text': 'text'}
    )
    assistant_msgs = df_pairs[['assistant_message_id', 'assistant_text']].rename(
        columns={'assistant_message_id': 'message_id', 'assistant_text': 'text'}
    )
    
    df_unique = pd.concat([user_msgs, assistant_msgs]).drop_duplicates('message_id')
    
    print(f"\n[sentiment_alignment] Extracted {len(df_unique):,} unique messages")
    
    return df_unique


def compute_sentiment(df_unique, model_name, device, batch_size, cache_dir, force_recompute):
    """Compute sentiment polarity for unique messages.
    
    Args:
        df_unique: DataFrame with unique messages
        model_name: Sentiment analysis model name
        device: Device to use (cpu/cuda)
        batch_size: Batch size for processing
        cache_dir: Directory for caching sentiment scores
        force_recompute: If True, ignore cached scores
        
    Returns:
        Dictionary mapping message_id to sentiment polarity [-1, 1]
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    sent_cache_path = cache_path / "message_sentiment.npy"
    ids_cache_path = cache_path / "message_ids_sentiment.npy"
    
    unique_message_ids = df_unique['message_id'].values
    unique_texts = df_unique['text'].tolist()
    
    print(f"\n[sentiment_alignment] Computing sentiment for {len(unique_texts):,} unique messages")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    
    # Check for cached sentiment
    if not force_recompute and sent_cache_path.exists() and ids_cache_path.exists():
        print("  Loading cached sentiment scores...")
        message_sentiments = np.load(sent_cache_path)
        message_ids_cached = np.load(ids_cache_path, allow_pickle=True)
        
        # Verify cache matches
        if (len(message_ids_cached) == len(unique_message_ids) and 
            np.array_equal(message_ids_cached, unique_message_ids)):
            print(f"  ✓ Loaded {len(message_sentiments):,} sentiment scores from cache")
            return dict(zip(message_ids_cached, message_sentiments))
        else:
            print("  Cache mismatch detected, recomputing...")
    
    # Detect device
    if device == "auto":
        try:
            import torch
            device_id = 0 if torch.cuda.is_available() else -1
        except ImportError:
            device_id = -1
    else:
        device_id = 0 if device == "cuda" else -1
    
    print(f"  Using device: {'cuda' if device_id >= 0 else 'cpu'}")
    
    # Load sentiment pipeline
    print(f"  Loading sentiment model...")
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device_id
    )
    
    # Compute sentiment in batches
    print(f"  Processing {len(unique_texts):,} texts (batch_size={batch_size})...")
    
    if device_id == -1:
        print("  NOTE: Running on CPU. This may take a while. Consider using GPU for faster processing.")
    
    scores = []
    for i in tqdm(range(0, len(unique_texts), batch_size), desc="Processing batches"):
        batch = unique_texts[i:i+batch_size]
        outputs = sentiment_pipe(batch, batch_size=batch_size, truncation=True)
        
        for out in outputs:
            label = str(out.get("label", "")).upper()
            score = float(out.get("score", 0.0))
            # Convert to polarity: positive score stays positive, negative becomes negative
            polarity = score if "POS" in label else -score
            scores.append(polarity)
    
    message_sentiments = np.array(scores, dtype="float32")
    
    # Save cache
    np.save(sent_cache_path, message_sentiments)
    np.save(ids_cache_path, unique_message_ids)
    
    print(f"  ✓ Saved sentiment scores to: {sent_cache_path}")
    print(f"    Mean polarity: {message_sentiments.mean():.4f}")
    print(f"    Std polarity: {message_sentiments.std():.4f}")
    
    return dict(zip(unique_message_ids, message_sentiments))


def compute_sentiment_similarity(df_pairs, sentiment_dict):
    """Compute sentiment similarity for turn pairs.
    
    Args:
        df_pairs: DataFrame with turn pairs
        sentiment_dict: Dictionary mapping message_id to sentiment polarity
        
    Returns:
        DataFrame with sentiment_similarity column added
    """
    print("\n[sentiment_alignment] Computing sentiment similarity...")
    
    # Look up sentiment for each message in pairs
    user_sent = np.array([sentiment_dict[mid] for mid in df_pairs['user_message_id']])
    assistant_sent = np.array([sentiment_dict[mid] for mid in df_pairs['assistant_message_id']])
    
    # Compute similarity: 1 - |difference|/2
    # This maps [-2, 0] (opposite to same) to [0, 1] (different to similar)
    sentiment_sim = 1.0 - (np.abs(user_sent - assistant_sent) / 2.0)
    
    df_pairs['user_sentiment'] = user_sent
    df_pairs['assistant_sentiment'] = assistant_sent
    df_pairs['sentiment_similarity'] = sentiment_sim
    
    print(f"  ✓ Computed sentiment similarity for {len(df_pairs):,} pairs")
    print(f"    Mean similarity: {sentiment_sim.mean():.4f}")
    print(f"    Std similarity: {sentiment_sim.std():.4f}")
    print(f"    Min similarity: {sentiment_sim.min():.4f}")
    print(f"    Max similarity: {sentiment_sim.max():.4f}")
    
    return df_pairs


def main():
    """Main execution function."""
    args = parse_args()
    
    # Setup paths
    output_path = Path(args.output).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("SENTIMENT ALIGNMENT COMPUTATION")
    print("=" * 70)
    
    # Load turn pairs
    if args.from_conversations:
        conversations_path = Path(args.from_conversations).resolve()
        df_pairs, df_messages = load_from_conversations(conversations_path)
    else:
        input_path = Path(args.input).resolve()
        df_pairs = load_turn_pairs(input_path)
    
    # Extract unique messages
    df_unique = extract_unique_messages(df_pairs)
    
    # Compute sentiment
    sentiment_dict = compute_sentiment(
        df_unique=df_unique,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=cache_dir,
        force_recompute=args.force_recompute
    )
    
    # Compute sentiment similarity
    df_pairs = compute_sentiment_similarity(df_pairs, sentiment_dict)
    
    # Save results
    print(f"\n[sentiment_alignment] Saving results to: {output_path}")
    df_pairs.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print("[sentiment_alignment] Done!")
    print(f"  Output: {output_path}")
    print(f"  Turn pairs: {len(df_pairs):,}")
    print("=" * 70)


if __name__ == "__main__":
    main()

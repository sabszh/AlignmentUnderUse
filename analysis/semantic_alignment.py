"""
Semantic alignment computation for ChatGPT conversation turn pairs.

This script computes semantic similarity between user and assistant messages using
sentence embeddings (all-mpnet-base-v2 from sentence-transformers). It creates
turn pairs in both directions (user→assistant and assistant→user) and computes
cosine similarity between their embeddings.

Processing steps:
1. Loads English conversations from JSONL
2. Extracts messages and creates unique message identifiers
3. Constructs turn pairs (user→assistant and assistant→user)
4. Computes or loads cached sentence embeddings for all unique messages
5. Computes semantic similarity (cosine) for each turn pair
6. Saves turn-level data with similarity scores

The script supports caching at multiple levels:
- Message embeddings (per unique message, reusable across runs)
- Similarity scores (per turn pair)

Usage:
    python -m analysis.semantic_alignment
    python -m analysis.semantic_alignment --input data/conversations_english.jsonl
    python -m analysis.semantic_alignment --output analysis/semantic_alignment.csv
    python -m analysis.semantic_alignment --force-recompute  # Ignore cached embeddings
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute semantic alignment for conversation turn pairs"
    )
    
    parser.add_argument(
        "--input",
        default="data/conversations_english.jsonl",
        help="Input JSONL file with English conversations (default: data/conversations_english.jsonl)",
    )
    
    parser.add_argument(
        "--output",
        default="analysis/semantic_alignment.csv",
        help="Output CSV file for turn-level semantic alignment (default: analysis/semantic_alignment.csv)",
    )
    
    parser.add_argument(
        "--embeddings-cache-dir",
        default="data",
        help="Directory for caching embeddings (default: data)",
    )
    
    parser.add_argument(
        "--model",
        default="all-mpnet-base-v2",
        help="Sentence transformer model name (default: all-mpnet-base-v2)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding computation (default: 256)",
    )
    
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of embeddings (ignore cache)",
    )
    
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for computation (default: auto)",
    )
    
    return parser.parse_args()


def load_messages(input_path):
    """Load and extract messages from conversations.
    
    Args:
        input_path: Path to input JSONL file
        
    Returns:
        DataFrame with columns: share_id, message_id, role, backend_index, text,
        plus conversation metadata and struct_* fields
    """
    print(f"[semantic_alignment] Loading conversations from: {input_path}")
    
    message_rows = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            obj = json.loads(line)
            share_id = obj.get('share_id')
            
            # Conversation-level metadata
            conversation_language = obj.get('conversation_language')
            conversation_metadata = obj.get('conversation_metadata') or {}
            default_model_slug = conversation_metadata.get('default_model_slug')
            conversation_title = conversation_metadata.get('title')
            conversation_create_time = conversation_metadata.get('create_time')
            
            if isinstance(conversation_create_time, (int, float)):
                conversation_create_time_iso = pd.to_datetime(conversation_create_time, unit='s', utc=True)
            else:
                conversation_create_time_iso = pd.NaT
            
            reddit_sources = obj.get('reddit_sources') or []
            subreddits = sorted({s.get('subreddit') for s in reddit_sources 
                               if isinstance(s, dict) and s.get('subreddit')})
            subreddits_joined = '|'.join(subreddits) if subreddits else None
            
            # Extract struct_* fields
            struct_fields = {k: v for k, v in obj.items() 
                           if isinstance(k, str) and k.startswith('struct_')}
            
            # Extract messages
            for msg in obj.get('messages', []):
                role = msg.get('role')
                if role not in {'user', 'assistant'}:
                    continue
                
                # Extract text
                extracted_text = None
                text = msg.get('text')
                if isinstance(text, str) and text.strip():
                    extracted_text = text
                else:
                    raw_content = msg.get('raw_content', {})
                    parts = raw_content.get('parts') if isinstance(raw_content, dict) else None
                    if isinstance(parts, list):
                        joined = ' '.join(p for p in parts if isinstance(p, str) and p.strip())
                        if joined.strip():
                            extracted_text = joined
                
                if extracted_text is None:
                    continue
                
                row = {
                    'share_id': share_id,
                    'conversation_title': conversation_title,
                    'conversation_create_time': conversation_create_time,
                    'conversation_create_time_iso': conversation_create_time_iso,
                    'conversation_language': conversation_language,
                    'default_model_slug': default_model_slug,
                    'subreddits': subreddits_joined,
                    'message_model_slug': msg.get('model_slug'),
                    'role': role,
                    'backend_index': msg.get('backend_index'),
                    'text': extracted_text
                }
                row.update(struct_fields)
                message_rows.append(row)
    
    df_messages = pd.DataFrame(message_rows)
    df_messages = df_messages.dropna(subset=['share_id', 'backend_index', 'text'])
    
    # Create unique message identifiers
    df_messages['message_id'] = (df_messages['share_id'] + '_' + 
                                 df_messages['backend_index'].astype(str) + '_' + 
                                 df_messages['role'])
    
    print(f"  Loaded {len(df_messages):,} messages from {df_messages['share_id'].nunique():,} conversations")
    
    return df_messages


def create_turn_pairs(df_messages):
    """Create turn pairs from messages.
    
    Args:
        df_messages: DataFrame with messages
        
    Returns:
        DataFrame with turn pairs, containing:
        - share_id, direction (user_to_assistant or assistant_to_user)
        - turn_index, user_text, assistant_text
        - user_message_id, assistant_message_id
        - conversation metadata
    """
    print("\n[semantic_alignment] Creating turn pairs...")
    
    pairs = []
    
    metadata_cols = [
        'conversation_title', 'conversation_create_time', 'conversation_create_time_iso',
        'conversation_language', 'default_model_slug', 'subreddits'
    ] + [c for c in df_messages.columns if c.startswith('struct_')]
    
    for share_id, group in df_messages.groupby('share_id'):
        group_sorted = group.sort_values('backend_index', kind='mergesort')
        rows = group_sorted.to_dict('records')
        ua_turn_index = 0
        au_turn_index = 0
        
        for i in range(len(rows) - 1):
            current_role = rows[i]['role']
            next_role = rows[i + 1]['role']
            
            if current_role == 'user' and next_role == 'assistant':
                ua_turn_index += 1
                assistant_model_slug = rows[i + 1].get('message_model_slug') or rows[i + 1].get('default_model_slug')
                pair = {
                    'share_id': share_id,
                    'direction': 'user_to_assistant',
                    'turn_index': ua_turn_index,
                    'assistant_model_slug': assistant_model_slug,
                    'user_text': rows[i]['text'],
                    'assistant_text': rows[i + 1]['text'],
                    'user_message_id': rows[i]['message_id'],
                    'assistant_message_id': rows[i + 1]['message_id']
                }
                for col in metadata_cols:
                    pair[col] = rows[i].get(col)
                pairs.append(pair)
                
            elif current_role == 'assistant' and next_role == 'user':
                au_turn_index += 1
                assistant_model_slug = rows[i].get('message_model_slug') or rows[i].get('default_model_slug')
                pair = {
                    'share_id': share_id,
                    'direction': 'assistant_to_user',
                    'turn_index': au_turn_index,
                    'assistant_model_slug': assistant_model_slug,
                    'user_text': rows[i + 1]['text'],
                    'assistant_text': rows[i]['text'],
                    'user_message_id': rows[i + 1]['message_id'],
                    'assistant_message_id': rows[i]['message_id']
                }
                for col in metadata_cols:
                    pair[col] = rows[i].get(col)
                pairs.append(pair)
    
    df_pairs = pd.DataFrame(pairs)
    
    print(f"  Created {len(df_pairs):,} turn pairs")
    print(f"    user→assistant: {(df_pairs['direction'] == 'user_to_assistant').sum():,}")
    print(f"    assistant→user: {(df_pairs['direction'] == 'assistant_to_user').sum():,}")
    
    return df_pairs


def compute_embeddings(df_messages, model_name, device, batch_size, cache_dir, force_recompute):
    """Compute or load cached sentence embeddings for unique messages.
    
    Args:
        df_messages: DataFrame with messages
        model_name: Sentence transformer model name
        device: Device to use (cpu/cuda)
        batch_size: Batch size for encoding
        cache_dir: Directory for caching embeddings
        force_recompute: If True, ignore cached embeddings
        
    Returns:
        Tuple of (message_embeddings array, message_ids array)
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    message_emb_path = cache_path / "message_embeddings.npy"
    message_ids_path = cache_path / "message_ids.npy"
    
    # Get unique messages
    df_unique = df_messages[['message_id', 'text']].drop_duplicates('message_id')
    unique_message_ids = df_unique['message_id'].values
    unique_texts = df_unique['text'].tolist()
    
    print(f"\n[semantic_alignment] Computing embeddings for {len(unique_texts):,} unique messages")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    
    # Check for cached embeddings
    if not force_recompute and message_emb_path.exists() and message_ids_path.exists():
        print("  Loading cached embeddings...")
        message_embeddings = np.load(message_emb_path)
        message_ids_cached = np.load(message_ids_path, allow_pickle=True)
        
        # Verify cache matches
        if (len(message_ids_cached) == len(unique_message_ids) and 
            np.array_equal(message_ids_cached, unique_message_ids)):
            print(f"  ✓ Loaded {len(message_embeddings):,} embeddings from cache")
            return message_embeddings, message_ids_cached
        else:
            print("  Cache mismatch detected, recomputing...")
    
    # Load model
    print(f"  Loading model: {model_name}...")
    
    # Detect device
    if device == "auto":
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
    
    print(f"  Using device: {device}")
    model = SentenceTransformer(model_name, device=device)
    
    # Compute embeddings
    print(f"  Encoding {len(unique_texts):,} texts (batch_size={batch_size})...")
    
    dim = 768  # MPNet embedding size
    message_embeddings = np.zeros((len(unique_texts), dim), dtype="float32")
    
    for i in range(0, len(unique_texts), batch_size):
        batch_texts = unique_texts[i:i+batch_size]
        
        emb_batch = model.encode(
            batch_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        message_embeddings[i:i+len(batch_texts)] = emb_batch.cpu().numpy()
        
        # Save checkpoint
        np.save(message_emb_path, message_embeddings)
        np.save(message_ids_path, unique_message_ids)
        
        if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= len(unique_texts):
            print(f"    Processed {min(i + batch_size, len(unique_texts)):,} / {len(unique_texts):,}")
    
    print(f"  ✓ Saved embeddings to: {message_emb_path}")
    
    return message_embeddings, unique_message_ids


def compute_similarity(df_pairs, message_embeddings, message_ids):
    """Compute semantic similarity for turn pairs.
    
    Args:
        df_pairs: DataFrame with turn pairs
        message_embeddings: Array of message embeddings
        message_ids: Array of message IDs (same order as embeddings)
        
    Returns:
        DataFrame with semantic_similarity column added
    """
    print("\n[semantic_alignment] Computing semantic similarity...")
    
    # Create lookup dictionary
    message_id_to_embedding = dict(zip(message_ids, message_embeddings))
    
    # Look up embeddings for each pair
    print("  Looking up embeddings for turn pairs...")
    user_emb = np.array([message_id_to_embedding[mid] for mid in df_pairs['user_message_id']])
    assistant_emb = np.array([message_id_to_embedding[mid] for mid in df_pairs['assistant_message_id']])
    
    # Compute cosine similarity (embeddings are normalized, so dot product = cosine)
    print("  Computing cosine similarity...")
    similarities = (user_emb * assistant_emb).sum(axis=1)
    
    df_pairs['semantic_similarity'] = similarities
    
    print(f"  ✓ Computed similarity for {len(df_pairs):,} pairs")
    print(f"    Mean similarity: {similarities.mean():.4f}")
    print(f"    Std similarity: {similarities.std():.4f}")
    print(f"    Min similarity: {similarities.min():.4f}")
    print(f"    Max similarity: {similarities.max():.4f}")
    
    return df_pairs


def main():
    """Main execution function."""
    args = parse_args()
    
    # Setup paths
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    cache_dir = Path(args.embeddings_cache_dir).resolve()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("SEMANTIC ALIGNMENT COMPUTATION")
    print("=" * 70)
    
    # Load messages
    df_messages = load_messages(input_path)
    
    # Create turn pairs
    df_pairs = create_turn_pairs(df_messages)
    
    # Compute embeddings
    message_embeddings, message_ids = compute_embeddings(
        df_messages=df_messages,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=cache_dir,
        force_recompute=args.force_recompute
    )
    
    # Compute similarity
    df_pairs = compute_similarity(df_pairs, message_embeddings, message_ids)
    
    # Save results
    print(f"\n[semantic_alignment] Saving results to: {output_path}")
    df_pairs.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print("[semantic_alignment] Done!")
    print(f"  Output: {output_path}")
    print(f"  Turn pairs: {len(df_pairs):,}")
    print("=" * 70)


if __name__ == "__main__":
    main()

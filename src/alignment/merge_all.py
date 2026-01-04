"""
Script to merge conversation, LSM, sentiment, semantic, and topics data for downstream analysis.

This script loads turn-level and conversation-level data, merges them into a single DataFrame, drops redundant columns, and saves the result as a CSV.


Arguments:
    --conv      Path to the conversations JSONL file (required)
    --lsm       Path to the LSM scores CSV file (required)
    --sentiment Path to the sentiment alignment CSV file (required)
    --semantic  Path to the semantic alignment CSV file (required)
    --lexsyn    Path to the lexical/syntactic alignment CSV file (required)
    --topics    Path to the topics CSV file (required)
    --output    Path to the output merged CSV file (required)

Example usage:
    python -m src.alignment.merge_all \
        --conv data/processed/conversations_english.jsonl \
        --lsm data/derived/lsm_scores.csv \
        --sentiment data/derived/sentiment_alignment.csv \
        --semantic data/derived/semantic_alignment.csv \
        --lexsyn data/derived/lexsyn_alignment.csv \
        --topics data/outputs/topics/conversations_with_topics.csv \
        --output data/outputs/merged.csv

The script expects the turn structure and cross-check logic to match between the sources for correct merging.
"""

import argparse
import pandas as pd
import json
import os

def estimate_tokens(text):
    if not isinstance(text, str):
        return 0
    return len(text.split())

def load_conv_df(jsonl_path):
    conv_rows = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            conv_id = obj.get('conv_id', obj.get('share_id'))
            meta = obj.get('conversation_metadata', {})
            title = meta.get('title', '')
            has_custom_instructions = obj.get('has_custom_instructions', meta.get('has_custom_instructions', ''))
            custom_gpt_used = obj.get('custom_gpt_used', meta.get('custom_gpt_used', ''))
            has_reasoning = meta.get('has_reasoning', '')
            tools_used = meta.get('tools_used', '')
            author = None
            subreddit = None
            reddit_sources = obj.get('reddit_sources') or []
            if reddit_sources and isinstance(reddit_sources, list):
                for s in reddit_sources:
                    if isinstance(s, dict):
                        if not author and s.get('author'):
                            author = s.get('author')
                        if not subreddit and s.get('subreddit'):
                            subreddit = s.get('subreddit')
            messages = obj.get('messages', [])
            filtered_msgs = []
            for m in messages:
                if not isinstance(m, dict):
                    continue
                role = m.get('role')
                text = m.get('text')
                if role in {'user', 'assistant'} and isinstance(text, str) and text.strip():
                    filtered_msgs.append(m)
            user_turn = 0
            asst_turn = 0
            for i in range(1, len(filtered_msgs)):
                prev_msg = filtered_msgs[i-1]
                curr_msg = filtered_msgs[i]
                prev_role = prev_msg.get('role')
                curr_role = curr_msg.get('role')
                if prev_role == curr_role:
                    continue
                if prev_role == 'user' and curr_role == 'assistant':
                    user_turn += 1
                    turn_idx = user_turn
                    user_msg = prev_msg
                    assistant_msg = curr_msg
                elif prev_role == 'assistant' and curr_role == 'user':
                    asst_turn += 1
                    turn_idx = -asst_turn
                    user_msg = curr_msg
                    assistant_msg = prev_msg
                else:
                    continue
                user_message = user_msg.get('text', '')
                assistant_message = assistant_msg.get('text', '')
                user_tokens = user_msg.get('n_tokens')
                if user_tokens is None:
                    user_tokens = estimate_tokens(user_message)
                assistant_tokens = assistant_msg.get('n_tokens')
                if assistant_tokens is None:
                    assistant_tokens = estimate_tokens(assistant_message)
                n_tokens_total = user_tokens + assistant_tokens
                model_slug = assistant_msg.get('model_slug', '')
                conv_rows.append({
                    'conv_id': conv_id,
                    'title': title,
                    'turn': turn_idx,
                    'n_tokens_total': n_tokens_total,
                    'user_cross_check_5': str(user_message)[:5] if isinstance(user_message, str) else '',
                    'assistant_cross_check_5': str(assistant_message)[:5] if isinstance(assistant_message, str) else '',
                    'author': author,
                    'subreddit': subreddit,
                    'has_custom_instructions': has_custom_instructions,
                    'custom_gpt_used': custom_gpt_used,
                    'model_slug': model_slug,
                    'has_reasoning': has_reasoning,
                    'tools_used': tools_used
                })
    return pd.DataFrame(conv_rows)

def main():

    parser = argparse.ArgumentParser(
        description="Merge conversation, LSM, sentiment, semantic, and topics data for downstream analysis."
    )
    parser.add_argument('--conv', metavar='PATH', default='data/processed/conversations_english.jsonl',
                        help='Path to conversations JSONL file (default: data/processed/conversations_english.jsonl)')
    parser.add_argument('--lsm', metavar='PATH', default='data/derived/lsm_scores.csv',
                        help='Path to LSM scores CSV file (default: data/derived/lsm_scores.csv)')
    parser.add_argument('--sentiment', metavar='PATH', default='data/derived/sentiment_alignment.csv',
                        help='Path to sentiment alignment CSV file (default: data/derived/sentiment_alignment.csv)')
    parser.add_argument('--semantic', metavar='PATH', default='data/derived/semantic_alignment.csv',
                        help='Path to semantic alignment CSV file (default: data/derived/semantic_alignment.csv)')
    parser.add_argument('--topics', metavar='PATH', default='data/outputs/conversations_with_topics.csv',
                        help='Path to topics CSV file (default: data/outputs/conversations_with_topics.csv)')
    parser.add_argument('--output', metavar='PATH', default='data/outputs/merged.csv',
                        help='Path to output merged CSV file (default: data/outputs/merged.csv)')
    parser.add_argument('--lexsyn', metavar='PATH', default='data/derived/lexsyn_alignment.csv',
                        help='Path to lexical/syntactic alignment CSV file (default: data/derived/lexsyn_alignment.csv)')
    args = parser.parse_args()

    print('Loading conv_df...')
    conv_df = load_conv_df(args.conv)
    print('Loaded conv_df shape:', conv_df.shape)

    print('Loading lsm_scores...')
    lsm_scores = pd.read_csv(args.lsm)
    print('Loading sentiment...')
    sentiment = pd.read_csv(args.sentiment)
    print('Loading semantic...')
    semantic = pd.read_csv(args.semantic)
    print('Loading topics...')
    topics = pd.read_csv(args.topics)
    print('Loading lexsyn_alignment...')
    lexsyn = pd.read_csv(args.lexsyn)

    merge_keys = ['conv_id', 'turn', 'user_cross_check_5', 'assistant_cross_check_5']
    merged = pd.merge(conv_df, lsm_scores, on=merge_keys, how='left', suffixes=('', '_lsm'))
    if set(merge_keys).issubset(sentiment.columns):
        merged = pd.merge(merged, sentiment, on=merge_keys, how='left', suffixes=('', '_sentiment'))
    else:
        merged = pd.merge(merged, sentiment, on=['conv_id', 'turn'], how='left', suffixes=('', '_sentiment'))
    if set(merge_keys).issubset(semantic.columns):
        merged = pd.merge(merged, semantic, on=merge_keys, how='left', suffixes=('', '_semantic'))
    else:
        merged = pd.merge(merged, semantic, on=['conv_id', 'turn'], how='left', suffixes=('', '_semantic'))
    if set(merge_keys).issubset(lexsyn.columns):
        merged = pd.merge(merged, lexsyn, on=merge_keys, how='left', suffixes=('', '_lexsyn'))
    else:
        merged = pd.merge(merged, lexsyn, on=['conv_id', 'turn'], how='left', suffixes=('', '_lexsyn'))
    if 'conv_id' in topics.columns:
        merged = pd.merge(merged, topics, on='conv_id', how='left', suffixes=('', '_topics'))
    elif 'share_id' in topics.columns:
        merged = pd.merge(merged, topics, left_on='conv_id', right_on='share_id', how='left', suffixes=('', '_topics'))
    print('Merged DataFrame shape:', merged.shape)

    # Drop duplicate or unnecessary columns
    drop_cols = [
        c for c in merged.columns if c.endswith('_lsm') or c.endswith('_sentiment') or c.endswith('_semantic') or c.endswith('_topics')
    ]
    # Keep only one set of message_id columns if duplicated
    for col in ['user_message_id', 'assistant_message_id']:
        dups = [c for c in merged.columns if c.startswith(col) and c != col]
        drop_cols.extend(dups)
    drop_cols = list(set(drop_cols))
    merged_clean = merged.drop(columns=[c for c in drop_cols if c in merged.columns], errors='ignore')

    # Save
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    merged_clean.to_csv(args.output, index=False)
    print(f'Saved merged data to {args.output}')

if __name__ == '__main__':
    main()

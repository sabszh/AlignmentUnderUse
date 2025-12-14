"""
Fetch Reddit comments containing ChatGPT share links from posts with share URLs.

Strategy:
1. Load posts from reddit_posts.jsonl
2. For each post with comments, fetch all comments via link_id
3. Extract comments containing ChatGPT share URLs
4. Save to reddit_comments.jsonl

Usage:
    python -m data_collection.collect_reddit_comments
    python -m data_collection.collect_reddit_comments --posts-file data/reddit_posts.jsonl
    python -m data_collection.collect_reddit_comments --max-posts 100 --dry-run
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Set

import pandas as pd
import requests
from tqdm import tqdm

from .arctic_shift_api import COMMENTS_API_URL, normalize_comment
from .io_utils import ensure_dir, write_jsonl


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Namespace with posts_file, output_dir, outfile, max_posts, dry_run, delay arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fetch comments from posts with ChatGPT share links"
    )

    parser.add_argument(
        "--posts-file",
        default="../data/reddit_posts.jsonl",
        help="Input JSONL file with Reddit posts (default: data/reddit_posts.jsonl)",
    )

    parser.add_argument(
        "--output-dir",
        default="../data",
        help="Output directory relative to data_collection (default: data)",
    )

    parser.add_argument(
        "--outfile",
        default="reddit_comments.jsonl",
        help="Output JSONL filename (default: reddit_comments.jsonl)",
    )

    parser.add_argument(
        "--max-posts",
        type=int,
        default=None,
        help="Maximum number of posts to process (default: all posts with comments)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write files, only count matches",
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API requests in seconds (default: 0.5)",
    )

    return parser.parse_args()


def load_existing_ids(path: Path) -> Set[str]:
    """Load existing comment IDs from disk to avoid duplicates.
    
    Args:
        path: Path to existing JSONL file.
    
    Returns:
        Set of seen comment IDs.
    """
    seen_ids: Set[str] = set()

    if not path.exists():
        return seen_ids

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            item_id = obj.get("id")
            if item_id:
                seen_ids.add(item_id)

    return seen_ids


def extract_share_urls(text: str) -> list:
    """Extract ChatGPT share URLs from text.
    
    Handles various URL formats:
    - https://chatgpt.com/share/xxx
    - http://chatgpt.com/share/xxx
    - chatgpt.com/share/xxx (no protocol)
    - www.chatgpt.com/share/xxx
    - chat.openai.com/share/xxx (legacy)
    
    Args:
        text: Text to search for share URLs.
    
    Returns:
        List of share URLs found (normalized to https://).
    """
    # Pattern matches:
    # - Optional protocol (https?://)
    # - Optional www.
    # - chatgpt.com or chat.openai.com
    # - /share/
    # - ID (alphanumeric, hyphens, underscores)
    pattern = r'(?:https?://)?(?:www\.)?(?:chatgpt\.com|chat\.openai\.com)/share/[\w-]+'
    
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    # Normalize URLs to include https://
    normalized = []
    for url in matches:
        if not url.startswith('http'):
            url = 'https://' + url
        normalized.append(url)
    
    return normalized


def main() -> None:
    """Main entry point.
    
    Workflow:
    1. Parse CLI arguments
    2. Load Reddit posts from JSONL
    3. Filter to posts with comments
    4. For each post, fetch comments via link_id
    5. Extract comments containing ChatGPT share URLs
    6. Write to JSONL with deduplication
    """
    args = parse_args()

    # Setup paths
    base_dir = Path(__file__).resolve().parent
    posts_path = base_dir / args.posts_file
    output_dir = base_dir / args.output_dir
    ensure_dir(str(output_dir))
    out_path = output_dir / args.outfile

    # Load existing comment IDs for deduplication
    seen_ids = load_existing_ids(out_path)

    print(f"[collect_reddit_comments] Loading posts from: {posts_path}")
    
    # Load posts with pandas
    df = pd.read_json(posts_path, lines=True)
    posts_with_comments = df[df['num_comments'] > 0].copy()
    
    # Limit posts if requested
    if args.max_posts:
        posts_with_comments = posts_with_comments.head(args.max_posts)
    
    print(f"[collect_reddit_comments] Total posts: {len(df)}")
    print(f"[collect_reddit_comments] Posts with comments: {len(posts_with_comments)}")
    print(f"[collect_reddit_comments] Previously seen comments: {len(seen_ids)}")
    print(f"[collect_reddit_comments] Output: {out_path}")
    
    if args.dry_run:
        print(f"[collect_reddit_comments] DRY RUN MODE - no files will be written")
    
    print(f"\nProcessing posts...\n")
    
    # Stats tracking
    total_comments_fetched = 0
    total_share_urls_found = 0
    total_new_comments = 0
    posts_processed = 0
    posts_with_shares = 0
    
    # Process each post
    for idx, post in tqdm(posts_with_comments.iterrows(), total=len(posts_with_comments), desc="Processing posts"):
        post_id = post['id']
        link_id = f"t3_{post_id}"
        
        # Fetch comments for this post
        try:
            response = requests.get(
                COMMENTS_API_URL,
                params={"link_id": link_id, "limit": "auto"}
            )
            
            if not response.ok:
                continue
            
            data = response.json()
            comments = data.get('data', [])
            total_comments_fetched += len(comments)
            
            # Find comments with share URLs
            post_had_shares = False
            for raw_comment in comments:
                body = raw_comment.get('body', '')
                share_urls = extract_share_urls(body)
                
                if not share_urls:
                    continue
                
                # Normalize comment
                normalized = normalize_comment(raw_comment)
                comment_id = normalized.get("id")
                
                # Skip duplicates
                if comment_id and comment_id in seen_ids:
                    continue
                
                # Add share URLs to comment data
                normalized['share_urls'] = share_urls
                normalized['source_post_id'] = post_id
                
                # Track and write
                if comment_id:
                    seen_ids.add(comment_id)
                
                total_share_urls_found += len(share_urls)
                total_new_comments += 1
                post_had_shares = True
                
                if not args.dry_run:
                    write_jsonl(str(out_path), normalized)
            
            if post_had_shares:
                posts_with_shares += 1
            
            posts_processed += 1
            
            # Rate limiting
            time.sleep(args.delay)
            
        except Exception as e:
            print(f"\nError processing post {post_id}: {e}")
            continue
    
    # Summary
    print(f"\n[collect_reddit_comments] Done!")
    print(f"  Posts processed: {posts_processed}")
    print(f"  Posts with share URLs in comments: {posts_with_shares}")
    print(f"  Total comments fetched: {total_comments_fetched}")
    print(f"  Total share URLs found: {total_share_urls_found}")
    print(f"  New comments written: {total_new_comments}")
    print(f"  Total unique comments: {len(seen_ids)}")
    
    if total_comments_fetched > 0:
        hit_rate = (total_share_urls_found / total_comments_fetched) * 100
        print(f"  Hit rate: {hit_rate:.2f}%")


if __name__ == "__main__":
    main()

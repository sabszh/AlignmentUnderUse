"""
Fetch Reddit posts containing ChatGPT share links from Arctic Shift API.

Usage:
    python -m data_collection.collect_reddit_posts
    python -m data_collection.collect_reddit_posts --output-dir data --outfile reddit_posts.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Set

from tqdm import tqdm

from .arctic_shift_api import fetch_chatgpt_share_posts, normalize_post
from .io_utils import ensure_dir, write_jsonl


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Namespace with output_dir, outfile, dry_run, max_pages, continue_from arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fetch Reddit posts with ChatGPT share links from Arctic Shift API"
    )

    parser.add_argument(
        "--output-dir",
        default="../data",
        help="Output directory relative to data_collection (default: data)",
    )

    parser.add_argument(
        "--outfile",
        default="reddit_posts.jsonl",
        help="Output JSONL filename (default: reddit_posts.jsonl)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write files, only count matches",
    )
    
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Maximum number of pages to fetch (default: 1, ~1000 posts per page)",
    )
    
    parser.add_argument(
        "--continue",
        dest="continue_from_existing",
        action="store_true",
        help="Continue from the last post in existing file (pagination)",
    )

    return parser.parse_args()


def load_existing_ids(path: Path) -> Set[str]:
    """Load existing post IDs from disk to avoid duplicates.
    
    Args:
        path: Path to existing JSONL file.
    
    Returns:
        Set of seen post IDs.
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


def get_last_timestamp(path: Path) -> int:
    """Get the created_utc of the last (oldest) post in the file.
    
    Args:
        path: Path to existing JSONL file.
    
    Returns:
        Unix timestamp of the last post, or None if file doesn't exist or is empty.
    """
    if not path.exists():
        return None
    
    last_timestamp = None
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                created_utc = obj.get("created_utc")
                if created_utc is not None:
                    last_timestamp = created_utc
            except json.JSONDecodeError:
                continue
    
    return last_timestamp


def main() -> None:
    """Main entry point.
    
    Workflow:
    1. Parse CLI arguments
    2. Load existing post IDs (for deduplication)
    3. Fetch posts from Arctic Shift API with pagination
    4. Normalize and deduplicate
    5. Write to JSONL
    """
    args = parse_args()

    # Setup output path
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / args.output_dir
    ensure_dir(str(output_dir))
    out_path = output_dir / args.outfile

    # Load existing IDs
    seen_ids = load_existing_ids(out_path)

    # Get pagination starting point
    before_timestamp = None
    if args.continue_from_existing:
        before_timestamp = get_last_timestamp(out_path)
        if before_timestamp:
            print(f"[collect_reddit_posts] Continuing from timestamp: {before_timestamp}")
        else:
            print(f"[collect_reddit_posts] No existing data found, starting fresh")

    print(f"[collect_reddit_posts] Output: {out_path}")
    print(f"[collect_reddit_posts] Previously seen: {len(seen_ids)} posts")
    print(f"[collect_reddit_posts] Fetching up to {args.max_pages} page(s) from Arctic Shift...")

    # Paginated fetching
    total_fetched = 0
    total_written = 0
    
    for page in range(args.max_pages):
        print(f"\n[collect_reddit_posts] Page {page + 1}/{args.max_pages}")
        print(f"  Before timestamp: {before_timestamp if before_timestamp else 'None (latest)'}")
        
        # Fetch posts
        raw_posts = fetch_chatgpt_share_posts(before=before_timestamp)
        
        if not raw_posts:
            print(f"  No more posts returned, stopping pagination")
            break
        
        print(f"  Fetched: {len(raw_posts)} posts")
        total_fetched += len(raw_posts)

        # Process and deduplicate
        written = 0
        for raw_post in tqdm(raw_posts, desc=f"  Processing page {page + 1}"):
            normalized = normalize_post(raw_post)
            post_id = normalized.get("id")

            # Skip duplicates
            if post_id and post_id in seen_ids:
                continue

            if post_id:
                seen_ids.add(post_id)

            written += 1
            total_written += 1

            if not args.dry_run:
                write_jsonl(str(out_path), normalized)
        
        print(f"  New posts written: {written}")
        
        # Update before_timestamp for next page (use last post's timestamp)
        if raw_posts:
            last_post = normalize_post(raw_posts[-1])
            before_timestamp = last_post.get("created_utc")
        
        # If we got fewer posts than expected, we've reached the end
        if len(raw_posts) < 1000:
            print(f"  Received fewer than 1000 posts, likely end of data")
            break

    print(f"\n[collect_reddit_posts] Done!")
    print(f"  Total fetched: {total_fetched} posts")
    print(f"  New written:   {total_written}")
    print(f"  Total unique:  {len(seen_ids)}")


if __name__ == "__main__":
    main()

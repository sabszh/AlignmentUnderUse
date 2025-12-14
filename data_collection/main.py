"""
Main pipeline orchestrator for data collection.

Runs three stages of data collection:
1. Collect Reddit posts with ChatGPT share links from Arctic Shift
2. Collect Reddit comments containing ChatGPT share links from posts
3. Collect conversation data from all discovered ChatGPT share URLs

Usage:
    python -m data_collection.main
    python -m data_collection.main --reddit-only
    python -m data_collection.main --comments-only
    python -m data_collection.main --conversations-only
    python -m data_collection.main --limit 50
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Namespace with stage control and pass-through arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run data collection pipeline"
    )
    
    parser.add_argument(
        "--reddit-only",
        action="store_true",
        help="Only run Reddit post collection stage",
    )
    
    parser.add_argument(
        "--comments-only",
        action="store_true",
        help="Only run Reddit comments collection stage",
    )
    
    parser.add_argument(
        "--conversations-only",
        action="store_true",
        help="Only run conversation collection stage",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of conversations to fetch",
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip already fetched)",
    )
    
    parser.add_argument(
        "--refresh-missing",
        action="store_true",
        help="With --resume, re-fetch conversations that failed or have no messages",
    )
    
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory (default: data)",
    )
    
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Max pages to fetch from Reddit (default: 1, ~1000 posts per page)",
    )
    
    parser.add_argument(
        "--continue",
        dest="continue_pagination",
        action="store_true",
        help="Continue Reddit pagination from last post in existing file",
    )
    
    parser.add_argument(
        "--max-comments-posts",
        type=int,
        default=None,
        help="Max posts to process for comments (default: all posts with comments)",
    )
    
    parser.add_argument(
        "--comments-delay",
        type=float,
        default=0.5,
        help="Delay between comment API requests in seconds (default: 0.5)",
    )
    
    return parser.parse_args()


def run_reddit_collection(
    output_dir: str,
    max_pages: int = 1,
    continue_pagination: bool = False
) -> bool:
    """Run Reddit post collection stage.
    
    Args:
        output_dir: Directory for output files.
        max_pages: Maximum number of pages to fetch.
        continue_pagination: Continue from last post in existing file.
    
    Returns:
        True if successful, False otherwise.
    """
    print("\n" + "=" * 60)
    print("STAGE 1: Collecting Reddit posts from Arctic Shift")
    print("=" * 60 + "\n")
    
    cmd = [
        sys.executable, "-m", "data_collection.collect_reddit_posts",
        "--output-dir", output_dir,
        "--outfile", "reddit_posts.jsonl",
        "--max-pages", str(max_pages)
    ]
    
    if continue_pagination:
        cmd.append("--continue")
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_comments_collection(
    output_dir: str,
    max_posts: int = None,
    delay: float = 0.5
) -> bool:
    """Run Reddit comments collection stage.
    
    Args:
        output_dir: Directory for input/output files.
        max_posts: Optional limit on number of posts to process.
        delay: Delay between API requests in seconds.
    
    Returns:
        True if successful, False otherwise.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Collecting Reddit comments with share links")
    print("=" * 60 + "\n")
    
    cmd = [
        sys.executable, "-m", "data_collection.collect_reddit_comments",
        "--posts-file", f"{output_dir}/reddit_posts.jsonl",
        "--output-dir", output_dir,
        "--outfile", "reddit_comments.jsonl",
        "--delay", str(delay)
    ]
    
    if max_posts:
        cmd.extend(["--max-posts", str(max_posts)])
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_conversation_collection(
    output_dir: str, 
    limit: int = None, 
    resume: bool = False,
    refresh_missing: bool = False
) -> bool:
    """Run conversation collection stage.
    
    Args:
        output_dir: Directory for input/output files.
        limit: Optional limit on number of conversations.
        resume: Whether to resume from previous run.
        refresh_missing: Whether to re-fetch failed conversations.
    
    Returns:
        True if successful, False otherwise.
    """
    print("\n" + "=" * 60)
    print("STAGE 3: Collecting ChatGPT conversations")
    print("=" * 60 + "\n")
    
    # Collect from both posts and comments
    cmd = [
        sys.executable, "-m", "data_collection.collect_conversations",
        "--input", f"{output_dir}/reddit_posts.jsonl", f"{output_dir}/reddit_comments.jsonl",
        "--output", f"{output_dir}/conversations.jsonl"
    ]
    
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    if resume:
        cmd.append("--resume")
    
    if refresh_missing:
        cmd.append("--refresh-missing")
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def main() -> None:
    """Main entry point.
    
    Orchestrates all collection stages based on CLI arguments.
    """
    args = parse_args()
    
    # Determine which stages to run
    run_reddit = not (args.comments_only or args.conversations_only)
    run_comments = not (args.reddit_only or args.conversations_only)
    run_conversations = not (args.reddit_only or args.comments_only)
    
    success = True
    
    # Stage 1: Reddit posts
    if run_reddit:
        success = run_reddit_collection(
            args.output_dir,
            max_pages=args.max_pages,
            continue_pagination=args.continue_pagination
        )
        if not success:
            print("\n[ERROR] Reddit post collection failed!")
            sys.exit(1)
    
    # Stage 2: Reddit comments
    if run_comments:
        success = run_comments_collection(
            args.output_dir,
            max_posts=args.max_comments_posts,
            delay=args.comments_delay
        )
        if not success:
            print("\n[ERROR] Reddit comments collection failed!")
            sys.exit(1)
    
    # Stage 3: Conversations
    if run_conversations:
        success = run_conversation_collection(
            args.output_dir,
            limit=args.limit,
            resume=args.resume,
            refresh_missing=args.refresh_missing
        )
        if not success:
            print("\n[ERROR] Conversation collection failed!")
            sys.exit(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    if run_reddit:
        reddit_path = Path(__file__).parent / args.output_dir / "reddit_posts.jsonl"
        if reddit_path.exists():
            print(f"Reddit posts: {reddit_path}")
    
    if run_comments:
        comments_path = Path(__file__).parent / args.output_dir / "reddit_comments.jsonl"
        if comments_path.exists():
            print(f"Reddit comments: {comments_path}")
    
    if run_conversations:
        conv_path = Path(__file__).parent / args.output_dir / "conversations.jsonl"
        if conv_path.exists():
            print(f"Conversations: {conv_path}")
    
    print()


if __name__ == "__main__":
    main()

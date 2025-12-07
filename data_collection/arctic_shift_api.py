"""
Arctic Shift API client for fetching Reddit posts containing ChatGPT share links.
"""

import requests
from typing import Any, Dict, List, Optional


API_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"


def fetch_chatgpt_share_posts(
    limit: str = "auto",
    before: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Fetch Reddit posts with ChatGPT share URLs from Arctic Shift API.
    
    Args:
        limit: Number of posts to fetch. "auto" for max allowed (typically 1000).
        before: Unix timestamp - only fetch posts created before this time.
                Use for pagination by passing the created_utc of the last post.
    
    Returns:
        List of raw post objects from Arctic Shift, ordered newest to oldest.
    """
    params = {
        "url": "https://chatgpt.com/share/",
        "limit": limit
    }
    
    if before is not None:
        params["before"] = before

    resp = requests.get(API_URL, params=params)
    resp.raise_for_status()
    payload = resp.json()

    return payload.get("data", [])


def normalize_post(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an Arctic Shift post to analysis-friendly format.
    
    Args:
        item: Raw post object from Arctic Shift.
    
    Returns:
        Normalized post dictionary with standardized fields.
    """
    return {
        "id": item.get("id"),
        "name": item.get("name"), 
        "subreddit": item.get("subreddit"),
        "author": item.get("author"),
        "created_utc": item.get("created_utc"),
        "title": item.get("title"),
        "score": item.get("score"),
        "num_comments": item.get("num_comments"),
        "url": item.get("url"),
        "permalink": f"https://reddit.com{item.get('permalink')}"
                      if item.get("permalink") else None
    }

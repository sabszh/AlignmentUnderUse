"""
Fetch ChatGPT conversation data from share links using the backend API.

Reads Reddit posts and comments from JSONL files, extracts share URLs,
fetches conversations from ChatGPT backend API, and writes structured
conversation data to JSONL.

Usage:
    python -m data_collection.collect_conversations
    python -m data_collection.collect_conversations --input data/reddit_posts.jsonl data/reddit_comments.jsonl
    python -m data_collection.collect_conversations --limit 10 --resume
"""

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from curl_cffi import requests
from tqdm import tqdm

from .io_utils import ensure_dir

BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Mobile Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Cache-Control": "max-age=0",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}


def build_headers() -> Dict[str, str]:
    """Build request headers with optional cookies from environment.
    
    Checks environment variables for authentication cookies:
    - CHATGPT_COOKIE: Full Cookie header value
    - CF_CLEARANCE: Cloudflare clearance token
    
    Returns:
        Dict with User-Agent and optional Cookie header.
    """
    headers = dict(BASE_HEADERS)
    
    env_cookie = os.environ.get("CHATGPT_COOKIE")
    cf_clearance = os.environ.get("CF_CLEARANCE")
    
    if env_cookie:
        headers["Cookie"] = env_cookie.strip()
    elif cf_clearance:
        headers["Cookie"] = f"cf_clearance={cf_clearance.strip()}"
    
    return headers


def extract_share_id(url: str) -> Optional[str]:
    """Extract share ID from a chatgpt.com/share URL.
    
    Args:
        url: Full URL like "https://chatgpt.com/share/abc123"
    
    Returns:
        Share ID ("abc123") or None if not found.
    """
    if not url:
        return None
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    for i, part in enumerate(parts):
        if part == "share" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def load_targets(inputs: List[Path]) -> Dict[str, Dict]:
    """Load share URLs and Reddit metadata from JSONL files.
    
    Handles both reddit posts and comments with different field structures.
    Deduplicates at two levels:
    1. Share IDs - same share URL from multiple sources gets one entry
    2. Reddit sources - same Reddit post/comment ID won't be added twice to a share
    
    Args:
        inputs: List of JSONL file paths containing Reddit posts/comments with share URLs.
    
    Returns:
        Dict mapping share_id to {"url": str, "reddit_sources": List[Dict]}.
        Each source Dict contains: type (post/comment), id, subreddit, author,
        created_utc, score, permalink, and type-specific fields (title, body, etc.).
    """
    targets: Dict[str, Dict] = {}
    
    for path in inputs:
        if not path.exists():
            continue
        
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Detect if this is a comment or post
                is_comment = "share_urls" in obj or "parent_id" in obj or "link_id" in obj
                
                # Extract URLs based on source type
                if is_comment:
                    # Comments have share_urls array
                    urls = obj.get("share_urls", [])
                else:
                    # Posts have single url or urls array
                    urls = obj.get("urls") or ([obj.get("url")] if obj.get("url") else [])
                
                for url in urls:
                    share_id = extract_share_id(url)
                    if not share_id:
                        continue
                    
                    # Deduplicate by share_id - first URL wins
                    entry = targets.setdefault(
                        share_id, {"url": url, "reddit_sources": []}
                    )
                    
                    # Build metadata based on source type
                    if is_comment:
                        reddit_source = {
                            "type": "comment",
                            "id": obj.get("id"),
                            "name": obj.get("name"),
                            "subreddit": obj.get("subreddit"),
                            "author": obj.get("author"),
                            "created_utc": obj.get("created_utc"),
                            "body": obj.get("body"),
                            "score": obj.get("score"),
                            "link_id": obj.get("link_id"),
                            "parent_id": obj.get("parent_id"),
                            "source_post_id": obj.get("source_post_id"),
                            "permalink": obj.get("permalink"),
                        }
                    else:
                        reddit_source = {
                            "type": "post",
                            "id": obj.get("id"),
                            "name": obj.get("name"),
                            "subreddit": obj.get("subreddit"),
                            "author": obj.get("author"),
                            "created_utc": obj.get("created_utc"),
                            "title": obj.get("title"),
                            "score": obj.get("score"),
                            "num_comments": obj.get("num_comments"),
                            "permalink": obj.get("permalink"),
                        }
                    
                    # Deduplicate by source ID (post or comment ID)
                    source_ids = {s["id"] for s in entry["reddit_sources"] if s.get("id")}
                    if reddit_source["id"] and reddit_source["id"] not in source_ids:
                        entry["reddit_sources"].append(reddit_source)
    
    return targets


def already_done(output_path: Path, refresh_missing: bool) -> Set[str]:
    """Load share IDs that have already been fetched.
    
    Used for --resume mode to skip previously fetched shares.
    
    Args:
        output_path: Path to output JSONL file.
        refresh_missing: If True, exclude shares that were fetched but have
            no messages (allows re-fetching failed attempts).
    
    Returns:
        Set of share IDs to skip.
    """
    seen: Set[str] = set()
    
    if not output_path.exists():
        return seen
    
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            share_id = obj.get("share_id")
            if not share_id:
                continue
            
            # Skip if refresh_missing and no messages
            if refresh_missing and not obj.get("messages"):
                continue
            
            seen.add(share_id)
    
    return seen


def extract_messages(api_response: Dict) -> Optional[List[Dict]]:
    """Extract and order messages from ChatGPT API response with full metadata preservation.
    
    DESIGN PRINCIPLE: Lossless first, interpretation later.
    
    Preserves ALL metadata from backend API without filtering or guessing importance.
    Adds convenience fields for common queries but keeps full raw structures.
    
    Args:
        api_response: JSON response from /backend-api/share/{id} endpoint.
    
    Returns:
        List of message dicts with:
        - Full raw metadata (raw_metadata, raw_content, author_metadata)
        - Convenience fields (model_slug, citations, etc.)
        - Interaction flags (is_visually_hidden, is_redacted, etc.)
        
        Sorted by create_time (oldest first).
        Returns None if parsing fails.
    """
    if not isinstance(api_response, dict):
        return None
    
    mapping = api_response.get("mapping")
    if not isinstance(mapping, dict):
        return None
    
    messages = []
    for node_id, node in mapping.items():
        if not isinstance(node, dict):
            continue
        
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        
        # Extract raw structures (PRESERVE EVERYTHING)
        content = message.get("content") or {}
        metadata = message.get("metadata") or {}
        author = message.get("author") or {}
        
        # Extract convenience fields for common queries
        content_type = content.get("content_type")
        role = author.get("role")
        author_name = author.get("name")
        
        # Text content from parts
        parts = content.get("parts")
        text = None
        if isinstance(parts, list):
            text = "\n\n".join(str(p) for p in parts if p is not None)
        
        # Code content
        code = content.get("text") if content_type == "code" else None
        language = content.get("language")
        
        # Reasoning/thoughts content (o1/o3 models)
        thoughts = content.get("thoughts")
        
        # Build message object with full preservation
        msg = {
            # Raw data (LOSSLESS - keep everything as-is)
            "raw_metadata": metadata,
            "raw_content": content,
            "author_metadata": author.get("metadata"),
            
            # Core identifiers
            "id": message.get("id"),
            "node_id": node_id,
            "parent": node.get("parent"),
            "children": node.get("children"),
            
            # Author info
            "role": role,
            "author_name": author_name,
            
            # Timestamps
            "create_time": message.get("create_time"),
            "update_time": message.get("update_time"),
            
            # Content
            "content_type": content_type,
            "text": text,
            
            # Message properties
            "status": message.get("status"),
            "weight": message.get("weight"),
            
            # Interaction flags (explicit visibility + turn semantics)
            "interaction_flags": {
                "is_visually_hidden": metadata.get("is_visually_hidden_from_conversation", False),
                "is_redacted": metadata.get("is_redacted", False),
                "end_turn": message.get("end_turn"),
                "recipient": message.get("recipient"),
                "channel": message.get("channel"),
            },
        }
        
        # Convenience fields (derived views - NOT replacements for raw data)
        # Add code/language if present
        if code:
            msg["code"] = code
        if language:
            msg["language"] = language
        
        # Add reasoning traces if present
        if thoughts:
            msg["thoughts"] = thoughts
        
        # Extract commonly used metadata fields (convenience accessors)
        if metadata.get("model_slug"):
            msg["model_slug"] = metadata["model_slug"]
        if metadata.get("gizmo_id"):
            msg["gizmo_id"] = metadata["gizmo_id"]
        if metadata.get("turn_exchange_id"):
            msg["turn_exchange_id"] = metadata["turn_exchange_id"]
        if metadata.get("shared_conversation_id"):
            msg["shared_conversation_id"] = metadata["shared_conversation_id"]
        if metadata.get("message_type"):
            msg["message_type"] = metadata["message_type"]
        if metadata.get("parent_id"):
            msg["parent_id"] = metadata["parent_id"]
        if metadata.get("citations"):
            msg["citations"] = metadata["citations"]
        if metadata.get("content_references"):
            msg["content_references"] = metadata["content_references"]
        if metadata.get("attachments"):
            msg["attachments"] = metadata["attachments"]
        if metadata.get("reasoning_status"):
            msg["reasoning_status"] = metadata["reasoning_status"]
        if metadata.get("finished_duration_sec"):
            msg["reasoning_duration_sec"] = metadata["finished_duration_sec"]
        if metadata.get("request_id"):
            msg["request_id"] = metadata["request_id"]
        
        # Tool execution results
        if metadata.get("aggregate_result"):
            result = metadata["aggregate_result"]
            msg["execution_status"] = result.get("status")
            msg["execution_output"] = result.get("final_expression_output")
        
        # Latency tracking (optional convenience field)
        if metadata.get("sonic_classification_result"):
            sonic = metadata["sonic_classification_result"]
            if sonic.get("latency_ms") is not None:
                msg["latency_ms"] = sonic["latency_ms"]
        
        messages.append(msg)
    
    # Sort by creation time
    messages.sort(key=lambda m: m.get("create_time") or 0)
    
    # Add backend ordering index after sort
    for backend_index, msg in enumerate(messages):
        msg["backend_index"] = backend_index
    
    return messages


def compute_conversation_summary(messages: Optional[List[Dict]]) -> Optional[Dict]:
    """Generate summary statistics for a conversation.
    
    Counts messages by role and computes temporal statistics.
    
    Args:
        messages: List of message dicts with 'role' and 'create_time' keys.
    
    Returns:
        Dict with keys:
        - message_count: Total number of messages
        - user_messages: Count of user messages
        - assistant_messages: Count of assistant messages
        - system_messages: Count of system messages
        - has_system_prompt: Boolean for presence of system messages
        - first_message_time: Unix timestamp of first message (or None)
        - last_message_time: Unix timestamp of last message (or None)
        - duration_seconds: Time between first and last message (or None)
        
        Returns None if messages is empty or None.
    """
    if not messages:
        return None
    
    role_counts = defaultdict(int)
    timestamps = []
    
    for msg in messages:
        if msg.get("role"):
            role_counts[msg["role"]] += 1
        
        if msg.get("create_time") is not None:
            timestamps.append(msg["create_time"])
    
    summary = {
        "message_count": len(messages),
        "user_messages": role_counts.get("user", 0),
        "assistant_messages": role_counts.get("assistant", 0),
        "system_messages": role_counts.get("system", 0),
        "has_system_prompt": role_counts.get("system", 0) > 0,
    }
    
    if timestamps:
        timestamps.sort()
        summary["first_message_time"] = timestamps[0]
        summary["last_message_time"] = timestamps[-1]
        summary["duration_seconds"] = timestamps[-1] - timestamps[0]
    else:
        summary["first_message_time"] = None
        summary["last_message_time"] = None
        summary["duration_seconds"] = None
    
    return summary


def fetch_share(share_id: str, timeout: int) -> Tuple[Optional[str], Dict]:
    """Fetch ChatGPT share from backend API.
    
    Makes GET request to https://chatgpt.com/backend-api/share/{id}
    and parses the response.
    
    Args:
        share_id: The share ID from the URL.
        timeout: Request timeout in seconds.
    
    Returns:
        Tuple of (error_message, result_dict).
        - error_message: None on success, error string on failure.
        - result_dict: Contains keys:
            - status_code: HTTP status code (or None if request failed)
            - error: Error message (if any)
            - api_response: Full JSON response (on success)
            - messages: Parsed message list (on success)
    """
    url = f"https://chatgpt.com/backend-api/share/{share_id}"
    
    try:
        resp = requests.get(url, headers=build_headers(), timeout=timeout, impersonate="chrome")
    except Exception as exc:
        return str(exc), {"status_code": None, "error": str(exc)}
    
    if resp.status_code != 200:
        error = f"HTTP {resp.status_code}"
        return error, {"status_code": resp.status_code, "error": error}
    
    try:
        data = resp.json()
    except Exception as exc:
        error = f"JSON decode error: {exc}"
        return error, {"status_code": resp.status_code, "error": error}
    
    messages = extract_messages(data)
    
    return None, {
        "status_code": resp.status_code,
        "api_response": data,
        "messages": messages,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Namespace with parsed arguments: input, output, limit, timeout,
        sleep, resume, refresh_missing, keep_raw.
    """
    parser = argparse.ArgumentParser(
        description="Fetch ChatGPT conversations from share links"
    )
    
    parser.add_argument(
        "--input",
        nargs="+",
        default=["../data/reddit_posts.jsonl", "..data/reddit_comments.jsonl"],
        help="JSONL files with Reddit posts/comments (default: data/reddit_posts.jsonl data/reddit_comments.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="../data/conversations.jsonl",
        help="Output JSONL file (default: data/conversations.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max shares to fetch (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="Request timeout in seconds (default: 15)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Sleep between requests in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip share IDs already in output file",
    )
    parser.add_argument(
        "--refresh-missing",
        action="store_true",
        help="With --resume, re-fetch entries without messages",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Include raw API response in output",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point.
    
    Orchestrates the scraping workflow:
    1. Load share URLs and Reddit metadata from input JSONL files
    2. Deduplicate by share ID
    3. Optionally skip already-fetched shares (--resume)
    4. Fetch each share from ChatGPT backend API
    5. Extract messages and compute conversation summaries
    6. Write structured JSONL output with Reddit context
    """
    args = parse_args()
    
    base_dir = Path(__file__).resolve().parent
    input_paths = [base_dir / Path(p) for p in args.input]
    output_path = base_dir / args.output
    
    ensure_dir(str(output_path.parent))
    
    # Load targets
    targets = load_targets(input_paths)
    share_ids = list(targets.keys())
    
    if args.limit:
        share_ids = share_ids[:args.limit]
    
    # Handle resume
    skip_ids: Set[str] = set()
    if args.resume:
        skip_ids = already_done(output_path, refresh_missing=args.refresh_missing)
    
    print(f"[collect_conversations] Found {len(share_ids)} unique share URLs")
    if skip_ids:
        print(f"[collect_conversations] Skipping {len(skip_ids)} already done")
    
    # Fetch shares
    fetched = 0
    with output_path.open("a", encoding="utf-8") as out_f:
        for share_id in tqdm(share_ids, desc="Fetching conversations"):
            if share_id in skip_ids:
                continue
            
            target = targets[share_id]
            url = target["url"]
            reddit_sources = target["reddit_sources"]
            
            # Fetch from API
            error, result = fetch_share(share_id, timeout=args.timeout)
            messages = result.get("messages")
            status_code = result.get("status_code")
            api_response = result.get("api_response")
            
            # Extract conversation-level metadata
            conv_metadata = {}
            if api_response:
                conv_metadata["title"] = api_response.get("title")
                conv_metadata["conversation_id"] = api_response.get("conversation_id")
                conv_metadata["create_time"] = api_response.get("create_time")
                conv_metadata["update_time"] = api_response.get("update_time")
                conv_metadata["is_public"] = api_response.get("is_public")
                conv_metadata["is_archived"] = api_response.get("is_archived")
                conv_metadata["current_node"] = api_response.get("current_node")
                
                # Shared conversation ID (cross-references messages)
                if api_response.get("shared_conversation_id"):
                    conv_metadata["shared_conversation_id"] = api_response["shared_conversation_id"]
                
                # Model information (full object, not just slug)
                model_info = api_response.get("model") or {}
                if model_info:
                    conv_metadata["model"] = model_info
                
                # Memory scope (if used)
                if api_response.get("memory_scope"):
                    conv_metadata["memory_scope"] = api_response["memory_scope"]
                
                # Moderation state and results
                if api_response.get("moderation_state"):
                    conv_metadata["moderation_state"] = api_response["moderation_state"]
                if api_response.get("moderation_results"):
                    conv_metadata["moderation_results"] = api_response["moderation_results"]
                
                # URL access controls
                if api_response.get("safe_urls"):
                    conv_metadata["safe_urls"] = api_response["safe_urls"]
                if api_response.get("blocked_urls"):
                    conv_metadata["blocked_urls"] = api_response["blocked_urls"]
                
                # Tool and feature flags
                if api_response.get("disabled_tool_ids"):
                    conv_metadata["disabled_tool_ids"] = api_response["disabled_tool_ids"]
                if api_response.get("default_model_slug"):
                    conv_metadata["default_model_slug"] = api_response["default_model_slug"]
                
                # Continuation and context editing
                if api_response.get("continue_conversation_url"):
                    conv_metadata["continue_conversation_url"] = api_response["continue_conversation_url"]
                if api_response.get("has_user_editable_context") is not None:
                    conv_metadata["has_user_editable_context"] = api_response["has_user_editable_context"]
                
                # Additional boolean flags
                if api_response.get("is_indexable") is not None:
                    conv_metadata["is_indexable"] = api_response["is_indexable"]
                if api_response.get("is_better_metatags_enabled") is not None:
                    conv_metadata["is_better_metatags_enabled"] = api_response["is_better_metatags_enabled"]
                if api_response.get("sugar_item_visible") is not None:
                    conv_metadata["sugar_item_visible"] = api_response["sugar_item_visible"]
                if api_response.get("is_study_mode") is not None:
                    conv_metadata["is_study_mode"] = api_response["is_study_mode"]
                
                # Analyze conversation features
                mapping = api_response.get("mapping") or {}
                conv_metadata["mapping_size"] = len(mapping)
                conv_metadata["root_node"] = api_response.get("current_node")
                
                # Custom instructions flag (redacted but detectable)
                has_custom_instructions = any(
                    node.get("message", {}).get("content", {}).get("content_type") == "model_editable_context"
                    for node in mapping.values()
                )
                if has_custom_instructions:
                    conv_metadata["has_custom_instructions"] = True
                
                # Tool usage detection
                tool_messages = [
                    node.get("message", {}).get("author", {}).get("name")
                    for node in mapping.values()
                    if node.get("message", {}).get("author", {}).get("role") == "tool"
                ]
                tools_used = [t for t in tool_messages if t]
                if tools_used:
                    conv_metadata["tools_used"] = list(set(tools_used))
                    conv_metadata["tool_call_count"] = len(tools_used)
                
                # File attachments detection
                all_attachments = []
                for node in mapping.values():
                    msg_attachments = node.get("message", {}).get("metadata", {}).get("attachments", [])
                    if msg_attachments:
                        all_attachments.extend(msg_attachments)
                
                if all_attachments:
                    conv_metadata["file_count"] = len(all_attachments)
                    file_types = list(set(att.get("mime_type") for att in all_attachments if att.get("mime_type")))
                    if file_types:
                        conv_metadata["file_types"] = file_types
                
                # Reasoning/thinking detection (o1/o3 models)
                reasoning_msgs = [
                    node.get("message", {})
                    for node in mapping.values()
                    if node.get("message", {}).get("content", {}).get("content_type") == "thoughts"
                ]
                if reasoning_msgs:
                    conv_metadata["has_reasoning"] = True
                    conv_metadata["reasoning_message_count"] = len(reasoning_msgs)
                    # Sum total thinking time
                    thinking_times = [
                        msg.get("metadata", {}).get("finished_duration_sec")
                        for msg in reasoning_msgs
                        if msg.get("metadata", {}).get("finished_duration_sec")
                    ]
                    if thinking_times:
                        conv_metadata["total_thinking_seconds"] = sum(thinking_times)
                
                # Code execution detection
                code_msgs = [
                    node.get("message", {})
                    for node in mapping.values()
                    if node.get("message", {}).get("content", {}).get("content_type") == "code"
                ]
                exec_msgs = [
                    node.get("message", {})
                    for node in mapping.values()
                    if node.get("message", {}).get("content", {}).get("content_type") == "execution_output"
                ]
                if code_msgs:
                    conv_metadata["code_block_count"] = len(code_msgs)
                if exec_msgs:
                    conv_metadata["code_execution_count"] = len(exec_msgs)
                
                # Citation usage
                msgs_with_citations = [
                    node.get("message", {})
                    for node in mapping.values()
                    if node.get("message", {}).get("metadata", {}).get("citations")
                ]
                if msgs_with_citations:
                    conv_metadata["citation_count"] = sum(
                        len(msg.get("metadata", {}).get("citations", []))
                        for msg in msgs_with_citations
                    )
                
                # Custom GPT detection
                gizmo_ids = [
                    node.get("message", {}).get("metadata", {}).get("gizmo_id")
                    for node in mapping.values()
                    if node.get("message", {}).get("metadata", {}).get("gizmo_id")
                ]
                if gizmo_ids:
                    conv_metadata["custom_gpt_used"] = True
                    conv_metadata["gizmo_ids"] = list(set(gizmo_ids))
                
                # Conversation structure
                branches = [
                    node
                    for node in mapping.values()
                    if node.get("children") and len(node.get("children", [])) > 1
                ]
                if branches:
                    conv_metadata["branch_count"] = len(branches)
            
            # Compute summary
            conversation_summary = compute_conversation_summary(messages)
            
            # Add visibility summary to metadata
            if messages:
                conv_metadata["visibility_summary"] = {
                    "total_messages": len(messages),
                    "hidden_messages": sum(
                        1 for m in messages
                        if m.get("interaction_flags", {}).get("is_visually_hidden")
                    ),
                    "redacted_messages": sum(
                        1 for m in messages
                        if m.get("interaction_flags", {}).get("is_redacted")
                    ),
                }
            
            # Timestamps
            fetched_at = int(time.time())
            fetched_at_iso = datetime.fromtimestamp(
                fetched_at, tz=timezone.utc
            ).isoformat()
            
            # Build output
            out_obj = {
                "share_id": share_id,
                "url": url,
                "fetched_at": fetched_at,
                "fetched_at_iso": fetched_at_iso,
                "fetch_success": messages is not None,
                "status_code": status_code,
                "error": error or result.get("error"),
                "reddit_sources": reddit_sources,
                "conversation_metadata": conv_metadata,
                "conversation_summary": conversation_summary,
                "messages": messages,
            }
            
            if args.keep_raw and api_response:
                out_obj["raw"] = api_response
            
            # Write
            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_f.flush()
            
            fetched += 1
            
            # Rate limiting
            if fetched % 10 == 0:
                time.sleep(args.sleep)
    
    print(f"\n[collect_conversations] Done. Wrote {fetched} records to {output_path}")


if __name__ == "__main__":
    main()

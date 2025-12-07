# AlignmentUnderUse

Research project investigating how ChatGPT is used in practice through shared conversation analysis.

## Overview

This project collects and analyzes publicly shared ChatGPT conversations from Reddit to understand real-world usage patterns, interaction styles, and alignment in practice. The data collection pipeline leverages two key data sources:

1. **Arctic Shift API** - Historical Reddit archive providing full-text search across Reddit posts
2. **ChatGPT Backend API** - OpenAI's share endpoint exposing rich conversation metadata

The pipeline has two stages:
1. **Reddit post collection** - Find Reddit posts containing ChatGPT share links using Arctic Shift
2. **Conversation collection** - Fetch full conversation data with comprehensive metadata from ChatGPT backend API

## Data Sources & APIs

### Arctic Shift Reddit Archive

**Endpoint:** `https://arctic-shift.photon-reddit.com/api/posts/search`

Arctic Shift provides historical Reddit data with full-text search capabilities. The API returns comprehensive post metadata including:
- Post content and metadata (author, score, comments, timestamps)
- Subreddit information
- Direct links to ChatGPT shares found in post content

**Pagination:** Results are paginated using the `search_after` parameter for efficient large-scale collection. Supports configurable `size` parameter for batch fetching.

### ChatGPT Backend API

**Endpoint:** `https://chatgpt.com/backend-api/share/{share_id}`

OpenAI's backend API exposes rich metadata for publicly shared conversations.

**Response Structure:**
- 23 top-level fields including conversation metadata, moderation state, and model configuration
- `mapping` object containing complete message tree with parent-child relationships
- Per-message metadata: role, timestamps, content, model version, tool usage, reasoning traces, citations, attachments

**Authentication:** Requires valid Cookie header for access. Extract from browser session (see Authentication section below).

**Cloudflare Protection:** The endpoint employs Cloudflare bot mitigation. Standard HTTP clients (e.g., `requests`) are blocked with `cf-mitigated` challenge. This pipeline uses `curl-cffi` with Chrome browser impersonation to bypass protection.

## Technical Requirements

### Dependencies

- Python 3.8+
- `curl-cffi` - **Critical dependency** for Cloudflare bypass
- `requests` - Used for Arctic Shift API
- `tqdm` - Progress bars

**Why curl-cffi?** The ChatGPT backend API uses Cloudflare bot detection that blocks standard Python HTTP libraries. `curl-cffi` provides browser impersonation (`impersonate="chrome"`) that successfully bypasses these protections. Without it, all conversation fetches return HTTP 403 errors.

### Installation

```bash
pip install -r requirements.txt
```

**Note:** `curl-cffi` requires `curl` library compiled with HTTP/2 support. On most systems, it installs automatically via pip.

### Authentication

To fetch ChatGPT conversations, you need a valid session cookie:

1. **Log into ChatGPT** in your browser (https://chatgpt.com)
2. **Open Developer Tools** (F12)
3. **Navigate to Application/Storage** → Cookies → https://chatgpt.com
4. **Copy the entire Cookie header:**
   - In Network tab, refresh page, click any request, copy full `Cookie:` header value
   - Should include: `__Secure-next-auth.session-token`, `__Host-next-auth.csrf-token`, `cf_clearance`, etc.

5. **Set environment variable:**

**PowerShell:**
```powershell
[System.Environment]::SetEnvironmentVariable("CHATGPT_COOKIE", "your_full_cookie_header", "Process")
```

**Bash:**
```bash
export CHATGPT_COOKIE="your_full_cookie_header"
```

**Or use `.env` file:**
```
CHATGPT_COOKIE=your_full_cookie_header
```

**Note:** Cookies expire periodically. If you get HTTP 403 errors, refresh your cookie.

## Usage

### Pipeline Execution

**Run full pipeline:**
```bash
python data_collection/main.py
```

**Resume interrupted collection:**
```bash
python data_collection/main.py --resume
```

**Refresh failed fetches:**
```bash
python data_collection/main.py --resume --refresh-missing
```

**Run individual stages:**
```bash
# Stage 1 only: Reddit posts
python data_collection/main.py --reddit-only

# Stage 2 only: Conversations (requires reddit_posts.jsonl)
python data_collection/main.py --conversations-only
```

**Limit collection for testing:**
```bash
python data_collection/main.py --limit 100
```

### Command-Line Options

**Main Pipeline (`main.py`):**
- `--reddit-only` - Only run Reddit post collection
- `--conversations-only` - Only run conversation collection
- `--limit N` - Limit conversations to fetch
- `--resume` - Resume from previous run
- `--refresh-missing` - Re-fetch failed attempts when resuming
- `--output-dir PATH` - Output directory (default: `data`)

**Reddit Collection (`collect_reddit_posts.py`):**
- `--output-dir PATH` - Output directory (default: `data`)
- `--outfile NAME` - Output filename (default: `reddit_posts.jsonl`)
- `--dry-run` - Count matches without writing

**Conversation Collection (`collect_conversations.py`):**
- `--input FILES` - Input JSONL files (default: `data/reddit_posts.jsonl`)
- `--output FILE` - Output file (default: `data/conversations.jsonl`)
- `--limit N` - Max conversations to fetch
- `--timeout N` - Request timeout in seconds (default: 15)
- `--sleep N` - Sleep between requests in seconds (default: 1.0)
- `--resume` - Skip already fetched shares
- `--refresh-missing` - Re-fetch failed attempts when resuming
- `--keep-raw` - Include raw API response in output

## Output Schema

The pipeline produces two JSONL files with comprehensive metadata.

### reddit_posts.jsonl

Each line contains metadata for one Reddit post with ChatGPT share link:

```json
{
  "id": "abc123",
  "name": "t3_abc123",
  "subreddit": "ChatGPT",
  "author": "username",
  "created_utc": 1234567890,
  "title": "Post title",
  "score": 42,
  "num_comments": 5,
  "url": "https://chatgpt.com/share/xyz",
  "permalink": "https://reddit.com/r/ChatGPT/comments/..."
}
```

### conversations.jsonl

Each line contains full conversation data with enhanced metadata:

```json
{
  "share_id": "xyz",
  "url": "https://chatgpt.com/share/xyz",
  "fetched_at": 1234567890,
  "fetched_at_iso": "2025-12-06T10:30:00Z",
  "fetch_success": true,
  "status_code": 200,
  "error": null,
  "reddit_sources": [
    {
      "id": "abc123",
      "subreddit": "ChatGPT",
      "author": "username",
      "created_utc": 1234567890,
      "title": "Post title",
      "score": 42,
      "num_comments": 5,
      "permalink": "https://reddit.com/r/..."
    }
  ],
  "conversation_metadata": {
    "title": "Conversation title",
    "conversation_id": "conv_123",
    "create_time": 1234567890.0,
    "update_time": 1234567900.0,
    "model": "gpt-4",
    "is_public": true,
    "is_archived": false,
    "current_node": "node_xyz",
    "memory_scope": "conversation",
    "has_custom_instructions": false,
    "tools_used": ["python", "dalle"],
    "tool_call_count": 5,
    "file_count": 2,
    "file_types": ["txt", "pdf"],
    "has_reasoning": true,
    "reasoning_message_count": 3,
    "total_thinking_seconds": 12.5,
    "code_block_count": 4,
    "code_execution_count": 2,
    "citation_count": 3,
    "custom_gpt_used": false,
    "gizmo_ids": [],
    "branch_count": 2
  },
  "messages": [
    {
      "id": "msg_123",
      "node_id": "node_xyz",
      "parent": "node_abc",
      "children": ["node_def"],
      "role": "user",
      "create_time": 1234567890.0,
      "update_time": 1234567890.0,
      "content_type": "text",
      "text": "Message content",
      "model_slug": "gpt-4-turbo",
      "citations": [],
      "attachments": [],
      "recipient": "all",
      "metadata": {},
      "reasoning": null,
      "request_id": null
    }
  ]
}
```

**Conversation Metadata Fields:**

- **Basic:** `title`, `conversation_id`, `create_time`, `update_time`
- **Model:** `model` (conversation-level model slug)
- **Privacy:** `is_public`, `is_archived`
- **Structure:** `current_node`, `memory_scope`, `branch_count`
- **Features:**
  - `has_custom_instructions` - Detects custom instruction presence (content redacted in shares)
  - `tools_used` - List of tool names used (e.g., `["python", "dalle", "file_search"]`)
  - `tool_call_count` - Total tool invocations
  - `file_count` - Number of files attached
  - `file_types` - List of file extensions
  - `has_reasoning` - Whether conversation includes reasoning/thinking traces
  - `reasoning_message_count` - Count of messages with reasoning
  - `total_thinking_seconds` - Sum of reasoning duration
  - `code_block_count` - Count of code blocks in messages
  - `code_execution_count` - Count of executed code blocks
  - `citation_count` - Total citations across messages
  - `custom_gpt_used` - Whether custom GPT was used
  - `gizmo_ids` - List of custom GPT IDs

**Message Fields:**

- **Tree Structure:** `id`, `node_id`, `parent`, `children`
- **Timing:** `create_time`, `update_time`
- **Content:** `role`, `content_type`, `text`, `model_slug`
- **Metadata:** `citations`, `attachments`, `recipient`, `metadata`, `reasoning`, `request_id`

**Message Roles:** `user`, `assistant`, `system`, `tool`

**Content Types:** `text`, `code`, `execution_output`, `multimodal_text`, `model_editable_context` (custom instructions)

## Data Quality & Deduplication

### Deduplication

The pipeline implements three levels of deduplication to ensure data integrity:

1. **Reddit Post Level** - Deduplication by post ID during Stage 1 collection
2. **Share ID Level** - Same ChatGPT share URL appearing in multiple Reddit posts is collected once
3. **Resume Protection** - `--resume` flag skips shares already in output file (idempotent operation)

No data loss occurs during deduplication - the `reddit_sources` array preserves all Reddit posts that referenced each share.

### Data Verification

Each conversation record includes:
- `fetch_success` - Boolean indicating successful fetch
- `status_code` - HTTP response code
- `error` - Error message if fetch failed
- `fetched_at` / `fetched_at_iso` - Collection timestamp for reproducibility

Failed fetches can be retried using `--refresh-missing` flag.

## Reproducibility

This project is designed for research reproducibility:

### Data Provenance

- **Arctic Shift API:** Historical Reddit archive provides consistent, queryable Reddit data
- **Timestamps:** All data includes collection timestamps (`fetched_at`, `create_time`, `update_time`)
- **Source Tracking:** Each conversation links back to Reddit posts via `reddit_sources` array
- **Idempotent Collection:** Resume capability allows interrupted runs to continue without duplicates

### Replication Steps

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Extract ChatGPT cookies:**
   - Log into https://chatgpt.com
   - Open browser DevTools (F12) → Network tab
   - Refresh page, select any request
   - Copy full `Cookie:` header value
   - Set `CHATGPT_COOKIE` environment variable

3. **Run collection:**
   ```bash
   # Full pipeline
   python data_collection/main.py
   
   # Or with limit for testing
   python data_collection/main.py --limit 100
   ```

4. **Resume if interrupted:**
   ```bash
   python data_collection/main.py --resume
   ```

### Rate Limiting

- **Arctic Shift API:** No documented rate limits, but pipeline uses pagination (`search_after`)
- **ChatGPT Backend API:** 
  - Default: 1 second sleep between requests (`--sleep 1.0`)
  - Increase if experiencing HTTP 429 errors: `--sleep 2.0`
  - Timeout per request: 15 seconds (configurable via `--timeout`)

### Known Limitations

- **Cookie Expiration:** ChatGPT cookies expire periodically; refresh if encountering HTTP 403 errors
- **Cloudflare Protection:** Requires `curl-cffi` with browser impersonation; standard HTTP clients fail
- **Custom Instructions:** Content is redacted in shared conversations, but `has_custom_instructions` flag detects presence
- **Deleted Shares:** Some share links may return 404 if conversation owner deleted them
- **Private Shares:** Shares made private after posting to Reddit become inaccessible

## Project Structure

```
data_collection/
├── main.py                      # Pipeline orchestrator
├── collect_reddit_posts.py      # Stage 1: Reddit post collection via Arctic Shift
├── collect_conversations.py     # Stage 2: Conversation fetching via ChatGPT backend API
├── arctic_shift_api.py          # Arctic Shift API client with pagination
├── io_utils.py                  # JSONL I/O utilities
└── data/                        # Output directory
    ├── reddit_posts.jsonl       # Reddit posts with ChatGPT share links
    └── conversations.jsonl      # Full conversation data with enhanced metadata
```

## License

See LICENSE file for details.

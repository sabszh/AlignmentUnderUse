# Alignment Under Use

Research project investigating how ChatGPT is used in practice through shared conversation analysis.

**Figure 1** Overview of Alignment Under Use pipeline
![image](data/img/figure_1.png)


## Overview

This project collects and analyzes publicly shared ChatGPT conversations from Reddit to understand real-world usage patterns, interaction styles, and alignment in practice. 

**Data Collection:** The pipeline leverages two key data sources:
1. **Arctic Shift API** - Historical Reddit archive providing full-text search across Reddit posts and comments
2. **ChatGPT Backend API** - OpenAI's share endpoint exposing rich conversation metadata

Collection occurs in three stages: Reddit post discovery, comment extraction, and conversation fetching with comprehensive metadata.

**Alignment Analysis:** The project computes semantic and sentiment alignment in human-AI conversations:
- **Semantic Alignment** - Measures meaning similarity between user and assistant messages using sentence embeddings (cosine similarity)
- **Sentiment Alignment** - Measures emotional tone alignment using sentiment analysis (polarity difference)
- **Visualization** - Generates plots analyzing alignment patterns across conversation dynamics, model versions, and message characteristics

## Data Sources & APIs

### Arctic Shift Reddit Archive

Arctic Shift provides historical Reddit data with full-text search capabilities.

**Endpoints:**
- Posts: `https://arctic-shift.photon-reddit.com/api/posts/search`
- Comments: `https://arctic-shift.photon-reddit.com/api/comments/search`

**Collection Strategy:** Two-phase approach:
1. Search for posts containing ChatGPT share URLs
2. For each post with comments, fetch all comments and extract share URLs

Testing shows ~2.6% of comments on ChatGPT-related posts contain share URLs, yielding thousands of additional conversations.

### ChatGPT Backend API

OpenAI's backend API exposes rich metadata for shared conversations.

**Endpoint:** `https://chatgpt.com/backend-api/share/{share_id}`

**Response includes:**
- Conversation metadata (title, model, timestamps)
- Complete message tree with parent-child relationships
- Per-message metadata (role, timestamps, content, model version)
- Tool usage, reasoning traces, citations, attachments

**Cloudflare Protection:** Requires `curl-cffi` with browser impersonation to bypass bot detection.

## Technical Requirements

### Dependencies

Data collection:
- `curl-cffi` - Cloudflare bypass (critical for ChatGPT API)
- `requests` - Arctic Shift API client
- `tqdm` - Progress bars
- `pandas` - Data processing

Data cleaning:
- `ftfy` - Text encoding fixes
- `regex` - Enhanced pattern matching
- `langid` - Language detection

Alignment analysis:
- `sentence-transformers` - Semantic embeddings (all-mpnet-base-v2)
- `transformers` - Sentiment analysis (distilbert-sst-2)
- `torch` - PyTorch backend for models
- `numpy` - Array operations and caching
- `matplotlib`, `seaborn` - Visualization

Optional:
- `presidio-analyzer`, `presidio-anonymizer` - PII detection (anonymization utility only)

**Why curl-cffi?** The ChatGPT backend API uses Cloudflare bot detection. `curl-cffi` provides browser impersonation that bypasses these protections.

### Installation

```bash
pip install -r requirements.txt
```

### Authentication

To fetch ChatGPT conversations, you need a valid session cookie from your browser:

1. Log into https://chatgpt.com
2. Open Developer Tools (F12) → Network tab
3. Refresh page, select any request
4. Copy full `Cookie:` header value
5. Set environment variable:

**PowerShell:**
```powershell
[System.Environment]::SetEnvironmentVariable("CHATGPT_COOKIE", "your_cookie", "Process")
```

**Bash:**
```bash
export CHATGPT_COOKIE="your_cookie"
```

**Note:** Cookies expire periodically. Refresh if you get HTTP 403 errors.
If you only have a Cloudflare clearance token, set `CF_CLEARANCE` instead of `CHATGPT_COOKIE`.

## Usage

Run all commands from the repository root so the `src` package is discoverable.

### Pipeline Execution

**Run full pipeline:**
```bash
python -m src.collection.main
```

**Resume interrupted collection:**
```bash
python -m src.collection.main --resume
```

**Refresh failed fetches:**
```bash
python -m src.collection.main --resume --refresh-missing
```

**Run individual stages:**
```bash
# Stage 1 only: Reddit posts
python -m src.collection.main --reddit-only

# Stage 2 only: Reddit comments (requires reddit_posts.jsonl)
python -m src.collection.main --comments-only

# Stage 3 only: Conversations (requires reddit_posts.jsonl and reddit_comments.jsonl)
python -m src.collection.main --conversations-only
```

**Limit collection for testing:**
```bash
# Limit conversations to fetch
python -m src.collection.main --limit 100

# Limit posts to process for comments
python -m src.collection.main --max-comments-posts 10
```

### Data Cleaning

**Clean and filter dataset:**
```bash
python -m src.processing.cleaning
```

The cleaning script applies minimal destructive text normalization:
- Fixes text encoding issues (mojibake, HTML entities) with `ftfy`
- Cleans markdown formatting while preserving code blocks as `[CODE_BLOCK_REMOVED]`
- Extracts structural features (turn counts, message lengths)
- Detects language with `langid` (fast, deterministic)
- Filters to English-only conversations

**Options:**
```bash
--input PATH                 # Input file (default: data/raw/conversations.jsonl)
--output PATH                # Output file (default: data/processed/conversations_english.jsonl)
--output-all-clean PATH      # Save all successful fetches before language filtering
--skip-language-filter       # Skip language detection
--skip-markdown-cleaning     # Skip markdown cleaning
```
**Figure 2** Overview of data collection and cleaning
![image](data/img/figure_2.png)

### Alignment Analysis

**Compute semantic alignment (sentence embeddings):**
```bash
python -m src.measures.semantic_alignment
```

Computes semantic similarity using all-mpnet-base-v2 model. Creates turn pairs (user→assistant and assistant→user) and computes cosine similarity between sentence embeddings.

**Compute sentiment alignment:**
```bash
python -m src.measures.sentiment_alignment
```

Computes sentiment similarity using distilbert sentiment model. Maps sentiment to [-1, 1] polarity and computes similarity as `1 - |difference|/2`.

**Options:**
```bash
# Semantic alignment
--model NAME                 # Sentence transformer model (default: all-mpnet-base-v2)
--batch-size N               # Batch size (default: 256)
--device auto|cpu|cuda       # Computation device (default: auto)
--force-recompute            # Ignore cached embeddings

# Sentiment alignment
--from-conversations PATH    # Load from conversations JSONL instead of semantic_alignment.csv
--model NAME                 # Sentiment model (default: distilbert-base-uncased-finetuned-sst-2-english)

```

### End-to-End Analysis Script

Run the full analysis pipeline (semantic, sentiment, LSM, optional topics) and archive outputs:

```bash
bash scripts/alignment_score_extraction.sh \
  data/processed/conversations_english.jsonl \
  data/derived \
  data/outputs \
  false \
  true
```

Arguments (all optional, shown in order):
- input JSONL path (default: `data/processed/conversations_english.jsonl`)
- derived output dir (default: `data/derived`)
- outputs dir (default: `data/outputs`)
- skip topics (`true` or `false`, default: `false`)
- verbose (`true` or `false`, default: `true`)

### Command-Line Options

**Main Pipeline (`src/collection/main.py`):**
- `--reddit-only` - Only run Reddit post collection
- `--comments-only` - Only run Reddit comments collection
- `--conversations-only` - Only run conversation collection
- `--limit N` - Limit conversations to fetch
- `--max-pages N` - Max pages to fetch for Reddit posts (default: 1)
- `--continue` - Continue Reddit post pagination from last post
- `--max-comments-posts N` - Limit posts to process for comments
- `--comments-delay N` - Delay between comment API requests (default: 0.5s)
- `--resume` - Resume from previous run
- `--refresh-missing` - Re-fetch failed attempts when resuming
- `--output-dir PATH` - Output directory (default: `data/raw`)

**Reddit Posts Collection (`src/collection/collect_reddit_posts.py`):**
- `--output-dir PATH` - Output directory (default: `data/raw`)
- `--outfile NAME` - Output filename (default: `reddit_posts.jsonl`)
- `--max-pages N` - Max pages to fetch (default: 1, ~1000 posts per page)
- `--continue` - Continue pagination from last post
- `--dry-run` - Count matches without writing

**Reddit Comments Collection (`src/collection/collect_reddit_comments.py`):**
- `--posts-file PATH` - Input JSONL with posts (default: `data/raw/reddit_posts.jsonl`)
- `--output-dir PATH` - Output directory (default: `data/raw`)
- `--outfile NAME` - Output filename (default: `reddit_comments.jsonl`)
- `--max-posts N` - Max posts to process for comments
- `--delay N` - Delay between API requests (default: 0.5s)
- `--dry-run` - Count matches without writing

**Conversation Collection (`src/collection/collect_conversations.py`):**
- `--input FILES` - Input JSONL files (default: `data/raw/reddit_posts.jsonl data/raw/reddit_comments.jsonl`)
- `--output FILE` - Output file (default: `data/raw/conversations.jsonl`)
- `--limit N` - Max conversations to fetch
- `--timeout N` - Request timeout in seconds (default: 15)
- `--sleep N` - Sleep between requests in seconds (default: 1.0)
- `--resume` - Skip already fetched shares
- `--refresh-missing` - Re-fetch failed attempts when resuming
- `--keep-raw` - Include raw API response in output

**Data Cleaning (`src/processing/cleaning.py`):**
- `--input PATH` - Input file (default: `data/raw/conversations.jsonl`)
- `--output PATH` - Output file (default: `data/processed/conversations_english.jsonl`)
- `--output-all-clean PATH` - Save all successful fetches before language filtering
- `--skip-language-filter` - Skip language detection
- `--skip-markdown-cleaning` - Skip markdown normalization

**Semantic Alignment (`src/measures/semantic_alignment.py`):**
- `--input PATH` - Input JSONL (default: `data/processed/conversations_english.jsonl`)
- `--output PATH` - Output CSV (default: `data/derived/semantic_alignment.csv`)
- `--embeddings-cache-dir PATH` - Directory for cached embeddings (default: `data/derived`)
- `--model NAME` - Sentence transformer model (default: `all-mpnet-base-v2`)
- `--batch-size N` - Batch size (default: 256)
- `--device auto|cpu|cuda` - Computation device (default: auto)
- `--force-recompute` - Ignore cached embeddings

**Sentiment Alignment (`src/measures/sentiment_alignment.py`):**
- `--input PATH` - Input CSV from semantic_alignment (default: `data/derived/semantic_alignment.csv`)
- `--from-conversations PATH` - Alternatively load from conversations JSONL
- `--conversations PATH` - Conversations JSONL to load text for missing columns
- `--output PATH` - Output CSV (default: `data/derived/sentiment_alignment.csv`)
- `--cache-dir PATH` - Directory for cached sentiment scores (default: `data/derived`)
- `--model NAME` - Sentiment model (default: `distilbert-base-uncased-finetuned-sst-2-english`)
- `--batch-size N` - Batch size (default: 64)
- `--device auto|cpu|cuda` - Computation device (default: auto)
- `--force-recompute` - Ignore cached sentiment scores

## Output Schema

The pipeline produces multiple JSONL files with comprehensive metadata.

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

### reddit_comments.jsonl

Each line contains metadata for one Reddit comment with ChatGPT share link(s):

```json
{
  "id": "def456",
  "name": "t1_def456",
  "subreddit": "ChatGPT",
  "author": "username",
  "created_utc": 1234567890,
  "body": "Comment text with https://chatgpt.com/share/xyz",
  "score": 10,
  "link_id": "t3_abc123",
  "parent_id": "t3_abc123",
  "permalink": "https://reddit.com/r/ChatGPT/comments/.../def456",
  "share_urls": ["https://chatgpt.com/share/xyz"],
  "source_post_id": "abc123"
}
```

## Data Folder Structure

To keep the repository clean while preserving a clear workflow, data artifacts are organized under `data/`:

- raw: source dumps collected from APIs
  - reddit_posts.jsonl, reddit_comments.jsonl, conversations.jsonl
- processed: cleaned, curated datasets ready for analysis
  - conversations_english.jsonl, anonymized_conversations.jsonl, df_pairs.csv
- derived: computed arrays and intermediate features
  - message_embeddings.npy, message_ids.npy, message_sentiment.npy, message_ids_sentiment.npy
  - semantic_alignment.csv, sentiment_alignment.csv, lsm_scores.csv
- outputs: analysis outputs and merged datasets
  - merged.csv (merged features from `merge_all.py`)
  - outputs/bayes: Bayesian model outputs
  - outputs/gamm: GAMM model outputs
  - outputs/other: misc analysis outputs
  - outputs/topics: topic modeling outputs

Only `data/README.md` is tracked in Git; all other files are ignored via `.gitignore`.

## Topic Modeling (Three Models)

Run the KeyNMF pipeline over user-only, assistant-only, and combined documents:

```bash
python -m src.measures.topic_modeling --input data/processed/conversations_english.jsonl --keywords 9 --plot
```

Outputs are saved to `data/outputs/topics/`. Common options:
- `--output-dir PATH` - Output directory (default: `data/outputs/topics`)
- `--topics N` - Topics per model (default: 30)
- `--max-chars-per-doc N` - Truncate long documents (default: 20000)

## Linguistic Style Matching (LSM)

Compute linguistic style matching scores between sequential user-assistant message pairs:

```bash
python -m src.measures.lsm_scoring
python -m src.measures.lsm_scoring --input data/processed/conversations_english.jsonl --output data/derived/lsm_scores.csv
```

LSM measures linguistic alignment across functional word categories: articles, prepositions, pronouns, auxiliary verbs, conjunctions, negations, and common adverbs. No filtering is applied; all conversations are processed.

Output is saved to `data/derived/lsm_scores.csv` with columns: `conv_id`, `turn`, `lsm_score`.

## Lexical + Syntactic Alignment (LexSyn)

Compute lexical (word overlap) and syntactic (POS tag overlap) alignment per turn pair:

```bash
python -m src.measures.lexsyn_alignment --input data/processed/conversations_english.jsonl --output data/derived/lexsyn_alignment.csv
```

Requires spaCy and the English model:
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## Combined Analysis

To merge topic assignments with sentiment and LSM outputs, use `src/alignment/merge_all.py` and write a combined CSV:

```bash
python -m src.alignment.merge_all \
  --conv data/processed/conversations_english.jsonl \
  --lsm data/derived/lsm_scores.csv \
  --sentiment data/derived/sentiment_alignment.csv \
  --semantic data/derived/semantic_alignment.csv \
  --lexsyn data/derived/lexsyn_alignment.csv \
  --topics data/outputs/topics/conversations_with_topics.csv \
  --output data/outputs/merged.csv
```

**Note:** The `share_urls` field contains all ChatGPT share URLs extracted from the comment body. The pattern matches various URL formats:
- `https://chatgpt.com/share/...`
- `http://chatgpt.com/share/...`
- `chatgpt.com/share/...` (no protocol)
- `www.chatgpt.com/share/...`
- `chat.openai.com/share/...` (legacy domain)

All URLs are normalized to `https://` in the output.

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
   - Set `CHATGPT_COOKIE` environment variable (or `CF_CLEARANCE` if you only have the Cloudflare token)

3. **Run collection:**
   ```bash
   # Full pipeline (posts + comments + conversations)
   python -m src.collection.main
   
   # Or with limits for testing
   python -m src.collection.main --limit 100 --max-comments-posts 10
   ```

4. **Resume if interrupted:**
   ```bash
   python -m src.collection.main --resume
   ```

5. **Clean and filter data:**
   ```bash
   python -m src.processing.cleaning
   ```

### Rate Limiting

- **Arctic Shift API:** No documented rate limits
- **ChatGPT Backend API:** 
  - Default: 1 second sleep between requests
  - Increase if experiencing HTTP 429 errors: `--sleep 2.0`
  - Timeout per request: 15 seconds (configurable via `--timeout`)

### Known Limitations

- **Cookie Expiration:** ChatGPT cookies expire periodically
- **Cloudflare Protection:** Requires `curl-cffi` with browser impersonation
- **Custom Instructions:** Content redacted in shared conversations
- **Deleted/Private Shares:** May return 404 if owner deleted or made private

## Project Structure

```
AlignmentUnderUse/
├── src/
│   ├── alignment/
│   │   ├── bayes_topic_alignment.Rmd # Bayesian topic alignment
│   │   ├── gamm_modeling.Rmd         # GAMM alignment analysis
│   │   └── merge_all.py              # Merge outputs for downstream analysis
│   ├── measures/
│   │   ├── topic_modeling.py        # Three-model KeyNMF pipeline
│   │   ├── semantic_alignment.py    # Semantic similarity (sentence embeddings)
│   │   ├── sentiment_alignment.py   # Sentiment similarity
│   │   └── lsm_scoring.py           # Linguistic style matching
│   ├── collection/
│   │   ├── arctic_shift_api.py      # Arctic Shift API client (posts + comments)
│   │   ├── collect_reddit_posts.py  # Stage 1: Reddit post collection
│   │   ├── collect_reddit_comments.py # Stage 2: Reddit comments collection
│   │   ├── collect_conversations.py # Stage 3: ChatGPT conversation fetching
│   │   └── main.py                  # Pipeline orchestrator (3 stages)
│   ├── processing/
│   │   ├── cleaning.py              # Text normalization + language filtering
│   │   └── anonymize.py             # Optional PII anonymization utility
│   ├── schemas/
│   │   └── turn.py                  # Turn-level schema constants
│   └── utils/
│       └── io_utils.py              # JSONL IO utilities
├── data/                            # Ignored in Git (except README.md)
│   ├── README.md                    # Tracked; documents data layout
│   ├── raw/                         # Source dumps collected from APIs
│   │   ├── reddit_posts.jsonl
│   │   ├── reddit_comments.jsonl
│   │   └── conversations.jsonl
│   ├── processed/                   # Cleaned datasets ready for analysis
│   │   ├── conversations_english.jsonl
│   │   └── anonymized_conversations.jsonl
│   ├── derived/                     # Computed arrays and intermediate features
│   │   ├── message_embeddings.npy
│   │   ├── message_ids.npy
│   │   ├── message_sentiment.npy
│   │   ├── message_ids_sentiment.npy
│   │   ├── semantic_alignment.csv
│   │   ├── sentiment_alignment.csv
│   │   └── lsm_scores.csv
│   └── outputs/
│       ├── merged.csv
│       ├── bayes/
│       │   └── bayes_topic_alignment_outputs/
│       │       ├── figures/
│       │       ├── diagnostics/
│       │       └── ppc/
│       ├── gamm/
│       │   ├── figures/
│       │   └── gamm_models/
│       ├── other/
│       └── topics/
│           ├── conversations_with_topics.csv
│           ├── topic_distributions.png
│           └── combined_measures.csv
└── scripts/
    └── alignment_score_extraction.sh # End-to-end alignment score extraction
```

## Data Availability & Ethics

### Analysis Pipeline

**All analysis is performed on raw data.** The data cleaning pipeline (`src/processing/cleaning.py`) performs text normalization and language filtering but does not remove or obscure any identifying information. This preserves the full semantic and structural content needed for discourse and alignment analysis.

### Data Release Policy

**The conversational dataset is not released.** Raw conversation data and source URLs are withheld for the following reasons:

1. **Privacy considerations:** While conversations were publicly shared by users, re-distribution without explicit consent may violate reasonable privacy expectations.
2. **Ethical considerations:** Individuals who shared conversations may not have anticipated research use or corpus aggregation.
3. **Terms of service:** Redistribution of ChatGPT conversation data may conflict with OpenAI's terms of service.

Privacy risk is managed at the **disclosure boundary**, not during internal computation.

### What Is Shared

This repository provides:
- **Data collection pipeline:** Complete code for replicating the collection process
- **Analysis code:** Data cleaning, feature extraction, and analytical methods
- **Anonymization utility:** Optional tool for PII removal (`src/processing/anonymize.py`)

The anonymization script is **not part of the analytical pipeline.** It is provided as an optional utility for:
- Inspecting or sharing individual conversation excerpts
- Creating demonstration examples
- Use by third parties on independently collected data

This tool does not provide complete privacy protection and should not be relied upon as a sole safeguard.

### Replication

Results can be replicated by:
1. Using the provided pipeline to collect newly shared conversations
2. Following the data cleaning and analysis procedures documented in this repository
3. Comparing findings with those reported in associated publications

The collection pipeline is deterministic given the same Reddit data source and time period.

### Ethical Responsibility

**Users of this pipeline are responsible for:**
- Ensuring compliance with relevant terms of service
- Obtaining necessary ethical approvals for their research
- Making independent determinations about data sharing and privacy protection
- Respecting the privacy and dignity of individuals whose conversations are analyzed

This repository provides tools for research but does not make claims about the ethical status of any particular use.

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

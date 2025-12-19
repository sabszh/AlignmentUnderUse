# Data Directory

## Data Not Included

The data files are **not published in this repository** due to:
- Privacy considerations for shared conversations
- Large file sizes
- Potential terms of service restrictions

## Acquiring the Data

For information on how to collect the dataset yourself, please refer to the [main project README](../README.md), which includes:
- Data collection pipeline setup
- Arctic Shift API usage for Reddit posts/comments
- ChatGPT backend API access for conversation retrieval
- Technical requirements and dependencies

## Expected Data Files

After running the collection pipeline, this directory should contain:
- `conversations.jsonl` - Raw collected conversations with metadata
- `conversations_english.jsonl` - English-only filtered conversations
- `reddit_posts.jsonl` - Reddit posts containing ChatGPT share links
- `reddit_comments.jsonl` - Reddit comments with share links
- `sample.jsonl` - Sample data for testing

See the main README for full documentation on the data schema and collection process.

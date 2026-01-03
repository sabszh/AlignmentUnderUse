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

## Folder Structure

After running the collection and analysis pipeline, this directory is organized as:

```
data/
├── raw/                          # Source dumps from APIs
│   ├── reddit_posts.jsonl
│   ├── reddit_comments.jsonl
│   └── conversations.jsonl
├── processed/                    # Cleaned, curated datasets
│   ├── conversations_english.jsonl
│   ├── anonymized_conversations.jsonl
│   └── df_pairs.csv
├── derived/                      # Computed arrays and features
│   ├── assistant_embeddings.npy
│   ├── user_embeddings.npy
│   ├── semantic_similarity.npy
│   ├── message_sentiment.npy
│   └── lsm_scores.csv
├── outputs/
│   ├── merged.csv                # All features merged for analysis (from merge_all.py)
│   ├── bayes/                    # Bayesian model outputs
│   │   └── bayes_topic_alignment_outputs/ # brms models, diagnostics, PPCs
│   │       ├── figures/
│   │       ├── diagnostics/
│   │       └── ppc/
│   ├── gamm/                     # GAMM model outputs
│   │   ├── gamm_models/          # .rds per metric
│   │   ├── figures/              # Saved plots
│   │   ├── gamm_summary.csv
│   │   └── gamm_smooths.csv
│   ├── other/                    # Misc analysis outputs
│   │   ├── clustering_stability_report.csv
│   │   └── topic_labels.csv
│   └── topics/                   # Topic modeling outputs
│       ├── conversations_with_topics.csv
│       ├── combined_measures.csv
│       └── topic_distributions.png
└── README.md
```

**Only `README.md` is tracked in Git.** All other files are ignored via `.gitignore`.

See the main README for full documentation on the data schema, collection, and merging process.

"""
Data cleaning script for ChatGPT conversations dataset.

PURPOSE: Text normalization and structural feature extraction.

This script performs REVERSIBLE cleaning that preserves analytical signal for:
- Alignment analysis
- Repair behavior detection
- Constraint-following assessment
- Discourse analysis

Processing steps:
1. Loads conversations from JSONL
2. Filters to successful fetches only
3. Cleans markdown formatting (MINIMAL - preserves structural signals)
   - Normalizes code blocks to typed placeholders:
     * [CODE_BLOCK_EXECUTABLE] - executable code
     * [FORMAL_SPEC_BLOCK] - formal specifications, schemas
     * [DATA_DUMP_BLOCK] - data dumps, logs
4. Derives structural features (turns, lengths, roles, block counts)
5. Detects language with confidence metadata
6. Filters to English-only conversations
7. Saves cleaned dataset ready for analysis

NOTE: Privacy protection (anonymization) is handled in a SEPARATE step.

Usage:
    python -m analysis.data_cleaning
    python -m analysis.data_cleaning --input data/conversations.jsonl --output data/conversations_clean.jsonl
    python -m analysis.data_cleaning --skip-markdown-cleaning  # Keep original formatting
"""

import argparse
import regex
import unicodedata
from pathlib import Path

import ftfy
import langid
import pandas as pd


def classify_block(block_text):
    """Classify a code block by content heuristics.
    
    Returns one of:
    - [CODE_BLOCK_EXECUTABLE] - contains executable code patterns
    - [FORMAL_SPEC_BLOCK] - contains formal specifications, schemas, diagrams
    - [DATA_DUMP_BLOCK] - data, logs, or other dumps
    
    Args:
        block_text: The text content of the code block
        
    Returns:
        str: Placeholder token for this block type
    """
    if not block_text:
        return '[DATA_DUMP_BLOCK]'
    
    # Normalize to lowercase for matching
    lower_text = block_text.lower()
    
    # Formal specification keywords (higher priority)
    formal_keywords = [
        'model', 'schema', 'uml', 'diagram', 'binding', 
        'acceptance criteria', 'entity', 'pipeline',
        'specification', 'constraint', 'invariant',
        'precondition', 'postcondition', 'ontology',
        'taxonomy', 'relationship', 'attribute',
    ]
    
    if any(keyword in lower_text for keyword in formal_keywords):
        return '[FORMAL_SPEC_BLOCK]'
    
    # Executable code keywords (programming languages)
    code_keywords = [
        'def ', 'class ', 'import ', 'from ', 'function',
        'const ', 'let ', 'var ', 'return',
        'select ', 'from ', 'where ', 'insert ',
        'public ', 'private ', 'void ', 'int ',
        'if (', 'for (', 'while (', 'switch (',
    ]
    
    # Structural code patterns
    code_patterns = ['{', '}', ';', '()', '=>', '->', '::',]
    
    # Check for code keywords
    if any(keyword in lower_text for keyword in code_keywords):
        return '[CODE_BLOCK_EXECUTABLE]'
    
    # Check for multiple code structural patterns
    pattern_count = sum(1 for pattern in code_patterns if pattern in block_text)
    if pattern_count >= 2:
        return '[CODE_BLOCK_EXECUTABLE]'
    
    # Default: treat as data dump
    return '[DATA_DUMP_BLOCK]'


def clean_markdown(text):
    """Remove markdown formatting from text with MINIMAL destruction of analytical signal.
    
    Design principle: Preserve structural/interactional cues, normalize presentation.
    
    Removes:
    - Headers (# ## ###)
    - Bold (**text** or __text__) - except when underscores are part of identifiers
    - Italic (*text* or _text_) - except snake_case and technical identifiers
    - Inline code (`code`)
    - Links ([text](url))
    - Lists (- item, * item, 1. item)
    - Tables
    - Horizontal rules (---, ***)
    - Blockquotes (> text)
    - Strikethrough (~~text~~)
    - ASCII art / box-drawing characters
    - Unicode formatting characters
    - Excessive punctuation patterns
    - Extra whitespace
    
    PRESERVES (normalizes with typed placeholders):
    - Code blocks (```...```) → Typed placeholders based on content:
      - [CODE_BLOCK_EXECUTABLE] - executable code (def, class, SELECT, etc.)
      - [FORMAL_SPEC_BLOCK] - formal specifications, schemas, UML diagrams
      - [DATA_DUMP_BLOCK] - data dumps, logs, other structured content
    
    Args:
        text: Input text with markdown formatting
        
    Returns:
        Clean text without markdown formatting but with structural signals preserved
    """
    if not isinstance(text, str) or not text:
        return text
    
    # NORMALIZE code blocks with typed placeholders (preserves structure)
    # Replace each fenced code block with a classifier-determined marker
    def replace_code_block(match):
        block_content = match.group(0)
        # Extract content between fences (remove ```language and closing ```)
        content = regex.sub(r'^```[a-zA-Z0-9]*\n?', '', block_content)
        content = regex.sub(r'\n?```$', '', content)
        return ' ' + classify_block(content) + ' '
    
    text = regex.sub(r'```[\s\S]*?```', replace_code_block, text)
    
    # Remove inline code (`code`) - extract content
    text = regex.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove headers (# ## ### etc)
    text = regex.sub(r'^#{1,6}\s+', '', text, flags=regex.MULTILINE)
    
    # Remove bold (**text** or __text__)
    text = regex.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    # Only remove __ when NOT part of snake_case/identifiers
    # Look for __text__ where text doesn't contain alphanumeric_underscore patterns
    text = regex.sub(r'__(?![a-zA-Z0-9_]+__)([^_]+)__', r'\1', text)
    
    # Remove italic (*text*)
    text = regex.sub(r'\*([^*]+)\*', r'\1', text)
    # Only remove _italic_ when NOT part of snake_case/identifiers
    # Don't remove underscores that are adjacent to alphanumeric characters
    # Pattern: _text_ where text doesn't look like code
    text = regex.sub(r'(?<![a-zA-Z0-9])_(?![a-zA-Z0-9_]+_)([^_\s]+)_(?![a-zA-Z0-9])', r'\1', text)
    
    # Remove strikethrough (~~text~~)
    text = regex.sub(r'~~([^~]+)~~', r'\1', text)
    
    # Remove links [text](url) -> keep text only
    text = regex.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove bare URLs
    text = regex.sub(r'https?://\S+', '', text)
    
    # Remove list markers (-, *, +, 1., 2., etc)
    text = regex.sub(r'^[\s]*[-*+]\s+', '', text, flags=regex.MULTILINE)
    text = regex.sub(r'^[\s]*\d+\.\s+', '', text, flags=regex.MULTILINE)
    
    # Remove blockquotes (> text)
    text = regex.sub(r'^>\s+', '', text, flags=regex.MULTILINE)
    
    # Remove horizontal rules (---, ***, ___)
    text = regex.sub(r'^[-*_]{3,}\s*$', '', text, flags=regex.MULTILINE)
    
    # Remove table formatting (| col1 | col2 |)
    text = regex.sub(r'\|[^\n]+\|', '', text)
    
    # Remove HTML tags (if any)
    text = regex.sub(r'<[^>]+>', '', text)
    
    # Remove ASCII art / box-drawing characters (┌─┐│└┘├┤┬┴┼╔═╗║╚╝╠╣╦╩╬▲▼◄►etc)
    # These are Unicode Box Drawing (U+2500-U+257F) and Block Elements (U+2580-U+259F)
    text = regex.sub(r'[\u2500-\u257F\u2580-\u259F\u25A0-\u25FF]+', '', text)
    
    # Remove excessive repeated punctuation patterns (====, ----, etc.)
    text = regex.sub(r'([=\-_~])\1{3,}', '', text)
    
    # NOTE: Emojis are PRESERVED (can be meaningful in sentiment/discourse analysis)
    # Uncomment line below ONLY if you need to remove emoji:
    # text = regex.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+', '', text)
    
    # Remove zero-width characters and other invisible Unicode
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cf')
    
    # Clean up whitespace
    text = regex.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    text = regex.sub(r' {2,}', ' ', text)  # Multiple spaces to single space
    text = regex.sub(r'^\s+|\s+$', '', text, flags=regex.MULTILINE)  # Trim lines
    text = text.strip()
    
    return text


def clean_message_text(messages, skip_cleaning=False):
    """Clean markdown formatting from all message text fields.
    
    Applies ftfy text fixing BEFORE markdown cleaning to handle encoding issues.
    
    Args:
        messages: List of message dictionaries
        skip_cleaning: If True, don't clean markdown
        
    Returns:
        Modified messages list with cleaned text
    """
    if not messages or skip_cleaning:
        return messages
    
    for msg in messages:
        if isinstance(msg, dict) and 'text' in msg and msg['text']:
            # First, fix text encoding issues (mojibake, HTML entities, etc.)
            msg['text'] = ftfy.fix_text(msg['text'])
            # Then, clean markdown formatting
            msg['text'] = clean_markdown(msg['text'])
    
    return messages


def detect_conversation_language(messages):
    """Detect dominant language in a conversation by combining all user+assistant text.
    
    Uses langid for faster, more accurate detection (especially on short texts).
    
    Returns tuple: (language_code, char_count, is_low_confidence, confidence_score)
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        tuple: (language_code: str, char_count: int, is_low_confidence: bool, confidence: float)
    """
    if not messages:
        return ('unknown', 0, True, 0.0)
    
    # Combine all user and assistant messages
    combined_text = []
    for msg in messages:
        if msg.get('role') in ['user', 'assistant'] and msg.get('text'):
            combined_text.append(msg['text'])
    
    if not combined_text:
        return ('unknown', 0, True, 0.0)
    
    # Combine into single text for more accurate detection
    full_text = ' '.join(combined_text)
    char_count = len(full_text.strip())
    
    if char_count < 20:
        return ('unknown', char_count, True, 0.0)
    
    # Use langid for language detection (faster and better on short texts than langdetect)
    lang_code, confidence = langid.classify(full_text)
    
    # Flag as low confidence if text is too short OR langid confidence is low
    is_low_confidence = char_count < 100 or confidence < 0.8
    
    return (lang_code, char_count, is_low_confidence, confidence)


def derive_structural_features(messages):
    """Derive privacy-safe structural features from cleaned messages.
    
    These features should be computed AFTER markdown cleaning to accurately
    count normalized block types. Text lengths are computed from cleaned text.
    
    Args:
        messages: List of message dictionaries (after markdown cleaning)
        
    Returns:
        dict: Structural features for the conversation
    """
    if not messages:
        return {
            'total_turns': 0,
            'user_turns': 0,
            'assistant_turns': 0,
            'system_turns': 0,
            'avg_user_message_length': 0.0,
            'avg_assistant_message_length': 0.0,
            'total_conversation_length': 0,
            'user_text_length': 0,
            'assistant_text_length': 0,
            'num_code_blocks': 0,
            'num_formal_blocks': 0,
            'num_data_blocks': 0,
            'has_any_block': False,
        }
    
    user_lengths = []
    assistant_lengths = []
    system_count = 0
    total_length = 0
    
    # Count block types across all messages
    num_code_blocks = 0
    num_formal_blocks = 0
    num_data_blocks = 0
    
    for msg in messages:
        role = msg.get('role', '')
        text = msg.get('text', '')
        text_len = len(text) if text else 0
        total_length += text_len
        
        # Count block types from cleaned text
        if text:
            num_code_blocks += text.count('[CODE_BLOCK_EXECUTABLE]')
            num_formal_blocks += text.count('[FORMAL_SPEC_BLOCK]')
            num_data_blocks += text.count('[DATA_DUMP_BLOCK]')
        
        if role == 'user':
            user_lengths.append(text_len)
        elif role == 'assistant':
            assistant_lengths.append(text_len)
        elif role == 'system':
            system_count += 1
    
    total_blocks = num_code_blocks + num_formal_blocks + num_data_blocks
    
    return {
        'total_turns': len(messages),
        'user_turns': len(user_lengths),
        'assistant_turns': len(assistant_lengths),
        'system_turns': system_count,
        'avg_user_message_length': sum(user_lengths) / len(user_lengths) if user_lengths else 0.0,
        'avg_assistant_message_length': sum(assistant_lengths) / len(assistant_lengths) if assistant_lengths else 0.0,
        'total_conversation_length': total_length,
        'user_text_length': sum(user_lengths),
        'assistant_text_length': sum(assistant_lengths),
        'num_code_blocks': num_code_blocks,
        'num_formal_blocks': num_formal_blocks,
        'num_data_blocks': num_data_blocks,
        'has_any_block': total_blocks > 0,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean and filter ChatGPT conversations dataset"
    )
    
    parser.add_argument(
        "--input",
        default="data/conversations.jsonl",
        help="Input JSONL file with conversations (default: data/conversations.jsonl)",
    )
    
    parser.add_argument(
        "--output",
        default="data/conversations_english.jsonl",
        help="Output JSONL file for cleaned English conversations (default: data/conversations_english.jsonl)",
    )
    
    parser.add_argument(
        "--output-all-clean",
        default=None,
        help="Optional output for all successful fetches (before language filtering)",
    )
    
    parser.add_argument(
        "--skip-language-filter",
        action="store_true",
        help="Skip language detection and filtering (only filter successful fetches)",
    )
    
    parser.add_argument(
        "--skip-markdown-cleaning",
        action="store_true",
        help="Skip markdown cleaning (keep original formatting)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup paths - resolve relative to cwd (where script is run from), not script location
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[data_cleaning] Loading data from: {input_path}")
    
    # Load data
    df = pd.read_json(input_path, lines=True)
    
    print(f"\nTotal records: {len(df)}")
    print(f"Successful fetches: {df['fetch_success'].sum()}")
    print(f"Failed fetches: {(~df['fetch_success']).sum()}")
    print(f"Success rate: {df['fetch_success'].mean():.2%}")
    
    # Filter to successful fetches
    df_clean = df[df['fetch_success']].copy()
    
    print(f"\nCleaned dataset: {len(df_clean)} successful conversations")
    print(f"Removed: {len(df) - len(df_clean)} failed fetches")
    
    # Clean markdown formatting from messages
    if args.skip_markdown_cleaning:
        print("\nSkipping markdown cleaning (--skip-markdown-cleaning enabled)")
    else:
        print("\nCleaning markdown formatting from messages...")
        print("(Removing headers, bold, links, tables, etc.)")
        print("(Normalizing: code blocks → [CODE_BLOCK_EXECUTABLE], [FORMAL_SPEC_BLOCK], [DATA_DUMP_BLOCK])")
        print("(Preserving: snake_case identifiers, emojis)")
        
        df_clean['messages'] = df_clean['messages'].apply(
            lambda msgs: clean_message_text(msgs, skip_cleaning=False)
        )
        
        print("Markdown cleaning complete!")
    
    # Derive structural features BEFORE anonymization
    print("\nDeriving structural features...")
    df_clean['structural_features'] = df_clean['messages'].apply(derive_structural_features)
    
    # Unpack structural features into top-level columns for easier analysis
    structural_df = pd.json_normalize(df_clean['structural_features'])
    for col in structural_df.columns:
        df_clean[f'struct_{col}'] = structural_df[col].values
    
    print(f"Structural features derived: {list(structural_df.columns)}")
    
    # Save all successful fetches if requested
    if args.output_all_clean:
        all_clean_path = Path(args.output_all_clean).resolve()
        all_clean_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_json(all_clean_path, orient='records', lines=True)
        print(f"\nSaved all successful fetches to: {all_clean_path}")
    
    # Language detection (unless skipped)
    if args.skip_language_filter:
        print("\nSkipping language detection (--skip-language-filter enabled)")
        df_final = df_clean
    else:
        print("\nDetecting language for each conversation...")
        print("(Using langid - fast and accurate on short texts)")
        
        # Detect language and get metadata
        lang_results = df_clean['messages'].apply(detect_conversation_language)
        
        df_clean['conversation_language'] = lang_results.apply(lambda x: x[0])
        df_clean['lang_detection_char_count'] = lang_results.apply(lambda x: x[1])
        df_clean['lang_detection_low_confidence'] = lang_results.apply(lambda x: x[2])
        df_clean['lang_detection_confidence'] = lang_results.apply(lambda x: x[3])
        
        # Language distribution
        lang_counts = df_clean['conversation_language'].value_counts()
        
        print("\nLanguage Distribution:")
        for lang, count in lang_counts.head(10).items():
            print(f"  {lang}: {count:,} ({count/len(df_clean)*100:.1f}%)")
        
        if len(lang_counts) > 10:
            print(f"  ... and {len(lang_counts) - 10} more languages")
        
        # Report low-confidence detections
        low_conf_count = df_clean['lang_detection_low_confidence'].sum()
        print(f"\nLow-confidence language detections: {low_conf_count:,} ({low_conf_count/len(df_clean)*100:.1f}%)")
        print(f"  (Conversations with < 100 characters of user+assistant text)")
        
        # Filter to English
        df_english = df_clean[df_clean['conversation_language'] == 'en'].copy()
        
        english_count = len(df_english)
        print(f"\nEnglish conversations: {english_count:,} ({english_count/len(df_clean)*100:.1f}%)")
        print(f"Non-English conversations: {len(df_clean) - english_count:,} ({(len(df_clean) - english_count)/len(df_clean)*100:.1f}%)")
        
        df_final = df_english
    
    # Save final dataset
    df_final.to_json(output_path, orient='records', lines=True)
    
    print(f"\n[data_cleaning] Done!")
    print(f"  Final dataset: {len(df_final)} conversations")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()

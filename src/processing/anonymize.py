"""
Minimal anonymization for ethical release of ChatGPT conversation data.

Uses Microsoft Presidio to anonymize only high-risk personally identifiable information:
- Person names (PERSON)
- Email addresses (EMAIL_ADDRESS)
- Phone numbers (PHONE_NUMBER)
- Precise street addresses (PRECISE_ADDRESS)

This is a post-processing step for ethical data sharing.

Usage:
    python -m src.processing.anonymize
    python -m src.processing.anonymize --input data/processed/conversations_english.jsonl --output data/processed/anonymized_conversations.jsonl
    python -m src.processing.anonymize --limit 100 --verbose
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from tqdm import tqdm


# Risk-based entity type -> placeholder mapping (Presidio entity types)
# Only anonymize HIGH-RISK personally identifying entities
ENTITY_PLACEHOLDERS = {
    # ALWAYS ANONYMIZE: Direct personal identifiers
    "PERSON": "[name]",
    "EMAIL_ADDRESS": "[email]",
    "PHONE_NUMBER": "[phone]",
    "PRECISE_ADDRESS": "[address]",
    
    # EXPLICITLY PRESERVED by default: All other entity types detected by Presidio
    # - ORGANIZATION, COMPANY (organizations, companies, institutions)
    # - LAW, LEGAL_DOCUMENT (legal documents, laws)
    # - PRODUCT, TECHNOLOGY (technologies, products, brands)
    # - EVENT (named events, historical events)
    # - LOCATION (countries, cities - usually general, not personally identifying)
    # - DATE_TIME (dates, times - usually general/historical)
    # - NUMERIC_VALUE, PERCENTAGE (numbers - important for analysis)
    # - URL, IP_ADDRESS (technical identifiers - PRESERVED, not PII in this context)
}

# High-risk entity types that are always anonymized
HIGH_RISK_ENTITIES = {
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "PRECISE_ADDRESS",  # Street-level addresses only
}


class AnonymizationStats:
    """Track anonymization statistics."""
    
    def __init__(self):
        self.total_records = 0
        self.total_texts_processed = 0
        self.total_replacements = 0
        self.entity_counts = defaultdict(int)
    
    def add_replacement(self, entity_type: str, count: int = 1):
        """Record a replacement."""
        self.total_replacements += count
        self.entity_counts[entity_type] += count
    
    def print_summary(self):
        """Print statistics summary."""
        print("\n" + "=" * 60)
        print("ANONYMIZATION SUMMARY")
        print("=" * 60)
        print(f"Total records processed: {self.total_records:,}")
        print(f"Total text fields processed: {self.total_texts_processed:,}")
        print(f"Total replacements made: {self.total_replacements:,}")
        
        if self.entity_counts:
            print("\nEntity Replacements:")
            for entity_type, count in sorted(self.entity_counts.items(), key=lambda x: -x[1]):
                placeholder = ENTITY_PLACEHOLDERS.get(entity_type, f"[{entity_type.lower()}]")
                print(f"  {entity_type:20} → {placeholder:15} : {count:,}")
        
        print()


def is_high_risk_entity(entity_type: str, text: str = None) -> bool:
    """Determine if a detected entity should be anonymized.
    
    Only anonymizes high-risk personally identifying information:
    - PERSON, EMAIL_ADDRESS, PHONE_NUMBER: Always anonymized
    - PRECISE_ADDRESS: Only if not a bare number (preserves years, counts)
    
    Args:
        entity_type: Presidio entity type
        text: The actual text of the entity
        
    Returns:
        True if entity should be anonymized
    """
    # High-risk entities are always anonymized
    if entity_type in HIGH_RISK_ENTITIES:
        # Special case: PRECISE_ADDRESS should not match bare numbers
        if entity_type == "PRECISE_ADDRESS" and text and text.strip().isdigit():
            return False
        return True
    
    # All other entities are preserved
    return False


def anonymize_text(text: str, analyzer: AnalyzerEngine, anonymizer: AnonymizerEngine, 
                   stats: AnonymizationStats = None, verbose: bool = False) -> str:
    """Anonymize high-risk PII in text using Microsoft Presidio.
    
    Anonymizes:
    - Person names (PERSON)
    - Email addresses (EMAIL_ADDRESS)
    - Phone numbers (PHONE_NUMBER)
    - Precise street addresses (PRECISE_ADDRESS)
    
    Preserves:
    - Organizations, products, laws, events
    - General locations (cities, countries)
    - Dates, times, years
    - Numbers, URLs, and other non-identifying content
    
    Args:
        text: Input text to anonymize
        analyzer: Presidio AnalyzerEngine instance
        anonymizer: Presidio AnonymizerEngine instance
        stats: Optional stats tracker
        verbose: Whether to print detailed progress
    
    Returns:
        Anonymized text with high-risk PII replaced by placeholders
    """
    if not isinstance(text, str) or not text:
        return text
    
    if stats:
        stats.total_texts_processed += 1
    
    # Analyze text with Presidio to detect entities
    analyzer_results = analyzer.analyze(
        text=text,
        language='en',
        entities=None  # Detect all, filter to high-risk below
    )
    
    if not analyzer_results:
        return text
    
    # Filter to only high-risk entities
    high_risk_results = []
    entity_type_counts = defaultdict(int)
    
    for result in analyzer_results:
        entity_text = text[result.start:result.end]
        
        if is_high_risk_entity(result.entity_type, entity_text):
            high_risk_results.append(result)
            entity_type_counts[result.entity_type] += 1
            
            if verbose:
                placeholder = ENTITY_PLACEHOLDERS.get(result.entity_type, f"[{result.entity_type.lower()}]")
                print(f"  Anonymizing {result.entity_type}: '{entity_text}' → {placeholder}")
    
    if not high_risk_results:
        return text
    
    # Record stats
    if stats:
        for entity_type, count in entity_type_counts.items():
            stats.add_replacement(entity_type, count)
    
    # Build operators dict for anonymization
    operators = {}
    for entity_type in entity_type_counts.keys():
        placeholder = ENTITY_PLACEHOLDERS.get(entity_type, f"[{entity_type.lower()}]")
        operators[entity_type] = OperatorConfig("replace", {"new_value": placeholder})
    
    # Anonymize text using Presidio
    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=high_risk_results,
        operators=operators
    )
    
    if verbose and high_risk_results:
        print(f"  → Made {len(high_risk_results)} replacements")
    
    return anonymized_result.text


def create_precise_address_recognizer():
    """Create a custom PatternRecognizer for precise street-level addresses.
    
    Detects re-identifiable address details while avoiding bare numbers:
    - Street name + house number (e.g., "Kongensvej 30", "Main St 12B")
    - Apartment/unit indicators with numbers
    - Postal codes with explicit context
    
    Does NOT match:
    - Bare numbers (years, counts, section numbers)
    - General locations (cities, countries)
    
    Returns:
        PatternRecognizer configured for PRECISE_ADDRESS detection
    """
    patterns = [
        # Street address with number (must have street indicator)
        Pattern(
            name="street_with_number",
            regex=r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Vej|Gade|Plads|Allé)\s+\d+[A-Za-z]?)\b",
            score=0.7,
        ),
        # Apartment/unit indicators (must have explicit indicator)
        Pattern(
            name="apartment_unit",
            regex=r"\b(?:Apt\.?|Apartment|Unit|Suite|Ste\.?)\s+\d+[A-Za-z]?\b",
            score=0.7,
        ),
        # House number + street (European format)
        Pattern(
            name="number_street",
            regex=r"\b\d+[A-Za-z]?\s+[A-Z][a-z]+(?:vej|gade|allé|plads|street|road|avenue|Street|Road|Avenue)\b",
            score=0.7,
        ),
        # Postal code with explicit context
        Pattern(
            name="postal_code_with_context",
            regex=r"\b(?:(?:ZIP|Postal Code|Postcode)\s*:?\s*\d{4,5}|DK-\d{4}|[A-Z][a-z]+\s+\d{4,5})\b",
            score=0.65,
        ),
    ]
    
    return PatternRecognizer(
        supported_entity="PRECISE_ADDRESS",
        patterns=patterns,
        context=["address", "live", "located", "street", "apartment", "zip", "postal"],
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Minimal anonymization for ethical release (removes high-risk PII only)"
    )
    
    parser.add_argument(
        "--input",
        default="data/processed/conversations_english.jsonl",
        help="Input JSONL file (default: data/processed/conversations_english.jsonl)",
    )
    
    parser.add_argument(
        "--output",
        default="data/processed/anonymized_conversations.jsonl",
        help="Output JSONL file (default: data/processed/anonymized_conversations.jsonl)",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to process (default: all)",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process data but don't write output file",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup paths
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[anonymize_data] Initializing Presidio...")
    
    # Initialize Presidio engines
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    
    # Register custom PRECISE_ADDRESS recognizer
    address_recognizer = create_precise_address_recognizer()
    analyzer.registry.add_recognizer(address_recognizer)
    
    print(f"[anonymize_data] Presidio loaded (with custom address recognizer)")
    
    print(f"\n[anonymize_data] Anonymization policy:")
    print(f"  Anonymizing: {', '.join(sorted(HIGH_RISK_ENTITIES))}")
    print(f"  Preserving: ORG, PRODUCT, LAW, LOCATION, DATE_TIME, numbers, URLs")
    
    print(f"\n[anonymize_data] Input: {input_path}")
    print(f"[anonymize_data] Output: {output_path}")
    
    if args.dry_run:
        print(f"[anonymize_data] DRY RUN MODE - no file will be written")
    
    if args.limit:
        print(f"[anonymize_data] Limiting to {args.limit} records")
    
    # Check input exists
    if not input_path.exists():
        print(f"\n[ERROR] Input file not found: {input_path}")
        return
    
    # Count lines for progress bar
    print(f"\n[anonymize_data] Counting records...")
    with input_path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    
    if args.limit:
        total_lines = min(total_lines, args.limit)
    
    print(f"[anonymize_data] Processing {total_lines:,} records...\n")
    
    # Initialize stats
    stats = AnonymizationStats()
    
    # Process file
    with input_path.open("r", encoding="utf-8") as fin:
        if not args.dry_run:
            fout = output_path.open("w", encoding="utf-8")
        
        for i, line in enumerate(tqdm(fin, total=total_lines, desc="Anonymizing")):
            if args.limit and i >= args.limit:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if args.verbose:
                    print(f"Warning: Failed to parse line {i+1}, treating as raw text")
                anonymized = anonymize_text(
                    line, analyzer, anonymizer, 
                    stats=stats, verbose=args.verbose
                )
                if not args.dry_run:
                    fout.write(json.dumps({"text": anonymized}, ensure_ascii=False) + "\n")
                stats.total_records += 1
                continue
            
            # Anonymize only message text fields
            if args.verbose and i % 100 == 0:
                print(f"\nProcessing record {i+1}/{total_lines}")
            
            # Only anonymize text within messages array
            if 'messages' in record and isinstance(record['messages'], list):
                for msg in record['messages']:
                    if isinstance(msg, dict) and 'text' in msg and msg['text']:
                        msg['text'] = anonymize_text(
                            msg['text'], analyzer, anonymizer, 
                            stats=stats, verbose=args.verbose
                        )
            
            if not args.dry_run:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            stats.total_records += 1
        
        if not args.dry_run:
            fout.close()
    
    # Print summaryWW
    stats.print_summary()
    
    if not args.dry_run:
        print(f"[anonymize_data] Output written to: {output_path}")
    else:
        print(f"[anonymize_data] Dry run complete - no file written")


if __name__ == "__main__":
    main()

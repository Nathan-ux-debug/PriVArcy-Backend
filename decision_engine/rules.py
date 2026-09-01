"""Regex-based PII detection rules for OCR'd text.

Deliberately simple, readable patterns rather than a full PII-detection
library — good enough to catch clearly-structured PII (card numbers, SSNs,
emails, phone numbers) in OCR output, with the understanding that OCR
text is noisy (misread characters, missing spaces) so these are
intentionally a bit permissive rather than pixel-perfect format validators.
"""

import re
from typing import List, Tuple

PII_PATTERNS = {
    # 13-19 digits, optionally grouped by spaces/dashes — covers most card
    # network formats without validating a real Luhn checksum (OCR text
    # is too noisy for that to be reliable).
    "credit_card_number": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    # US Social Security Number format: 123-45-6789
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),
    # Loosely matches US-style phone numbers with optional country code.
    "phone_number": re.compile(r"\b(?:\+?1[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b"),
    # Passport-like: 1-2 letters followed by 6-9 digits.
    "passport_like": re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
}


def contains_pii(text: str) -> Tuple[bool, List[str]]:
    """Check OCR'd text against every PII pattern.

    Returns:
        (True/False whether ANY pattern matched, list of pattern names that matched).
    """
    if not text:
        return False, []

    matched = [name for name, pattern in PII_PATTERNS.items() if pattern.search(text)]
    return (len(matched) > 0), matched

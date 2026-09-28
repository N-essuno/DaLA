"""Text-only necessary conditions shared by parsing and source curation."""
import re


def source_text_rejection(text, curation):
    if not 35 <= len(text) <= curation['max_sentence_chars']:
        return 'length'
    if not text[0].isupper() or text[-1] != '.':
        return 'fragment_or_format'
    if re.search(r'\b[^\W\d_]\.$', text) and text[-2].isupper():
        return 'truncated_initial_reference'
    if re.search(r'\s+[.,;:!?]|[,;:!?][^\W\d_]', text):
        return 'punctuation_spacing'
    if any(re.search(pattern, text) for pattern in curation.get('source_risk_patterns', [])):
        return 'source_extraction_risk'
    if re.search(r'[\[\]<>|~\\\n\r\u00AD\uFFFD]|https?://|www\.|[a-zà-öø-ÿ][A-Z]|\d[^\W\d_]', text):
        return 'source_extraction_risk'
    return None

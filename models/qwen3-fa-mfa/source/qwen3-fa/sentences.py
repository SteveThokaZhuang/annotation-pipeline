"""Map punctuation-delimited sentences onto exactly the Qwen output units."""
import re


def sentence_spans(text, language, processor):
    # Keep decimal points; include trailing quotes in the sentence.
    pieces = re.split(r'(?<=[。！？!?])\s*|(?<!\d)\.(?!\d)[\"”’]*\s*|\n+', text)
    pieces = [p.strip() for p in pieces if p.strip()]
    all_tokens = processor.encode_timestamp(text, language)[0]
    spans, combined = [], []
    for piece in pieces:
        tokens = processor.encode_timestamp(piece, language)[0]
        if tokens:
            spans.append(dict(start_index=len(combined), end_index=len(combined)+len(tokens), text=piece))
            combined.extend(tokens)
    if combined != all_tokens:
        # Conservative fallback: never guess token indices or truncate context.
        return [dict(start_index=0, end_index=len(all_tokens), text=text, context='whole_chunk_tokenization_fallback')]
    return spans

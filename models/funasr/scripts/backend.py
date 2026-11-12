"""Compatibility adapter for FunASR batched decoding with Transformers 4.57.

Upstream supplies a fixed prefill position_ids tensor to generate(). Transformers
extends the attention mask during decoding but does not extend supplied positions,
so those positions freeze at the prompt's last position after the first token.
Let Transformers derive positions from the growing attention mask on every step.
This also handles left padding correctly and leaves upstream source untouched.
"""
from functools import wraps


def install_generation_fix(llm):
    original = llm.generate
    stats = []

    @wraps(original)
    def generate(*args, **kwargs):
        stats.clear()
        if kwargs.get('attention_mask') is not None:
            kwargs.pop('position_ids', None)
        result = original(*args, **kwargs)
        eos = llm.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        for row in result.tolist():
            end = next((i for i, token in enumerate(row) if token in eos), None)
            stats.append(dict(generated_tokens=len(row) if end is None else end + 1,
                              finish_reason='length' if end is None else 'eos'))
        return result

    llm.generate = generate
    return stats

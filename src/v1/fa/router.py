"""Gold-free routing. The policy is fitted on one meeting, then frozen."""
from common import norm,structural,match_words

def features(text, words, duration):
    tokens=[norm(t) for t in text.split()]
    issues=structural(words,duration)
    lexical=match_words([{'text':t} for t in text.split()],words)
    return {
        'hard':float(any(k in {'zero_duration','negative_duration','nonfinite','out_of_bounds','nonmonotonic'} for _,k in issues) or bool(lexical['deleted'] or lexical['substituted'] or lexical['inserted'])),
        'short_ratio':sum(k=='short_duration' for _,k in issues)/max(len(tokens),1),
        'long':float(any(k=='long_duration' for _,k in issues)),
        'fillers':sum(t in {'uh','um','erm','er','hmm','mm'} for t in tokens)/max(len(tokens),1),
        'repeat':float(any(x==y for x,y in zip(tokens,tokens[1:]))),
        'gap':max([max(0,b['start']-a['end']) for a,b in zip(words,words[1:])]+[0]),
    }

def route(text, qwen_words, duration, policy):
    f=features(text,qwen_words,duration)
    for rule in policy['rules']:
        if f[rule['feature']] >= rule['threshold']:
            return rule['backend'],rule['feature']
    return 'qwen','default'

def acceptable(text, words, duration):
    if not words: return False
    m=match_words([{'text':t} for t in text.split()],words)
    return not (m['deleted'] or m['substituted'] or m['inserted']) and not any(k in {'zero_duration','negative_duration','nonfinite','out_of_bounds','nonmonotonic','overlap'} for _,k in structural(words,duration))

def choose(text, duration, candidates, policy):
    """Apply configured route, then validate fallback. Return review when none is sound."""
    q=candidates.get('qwen',{'words':[]})
    backend,reason=route(text,q.get('words',[]),duration,policy)
    if backend=='qwen' and not q.get('error'):
        return dict(q,selected_backend='qwen',route_reason=reason,needs_review=not acceptable(text,q['words'],duration))
    target=candidates.get(backend)
    if target and not target.get('error') and acceptable(text,target.get('words',[]),duration) and (not policy.get('guarded') or consistent_with_primary(q.get('words',[]),target['words'])):
        return dict(target,selected_backend=backend,route_reason=reason,needs_review=False)
    # Invalid fallback is never repaired by stretching or fabricating word timestamps.
    return dict(q,selected_backend='qwen',route_reason=reason+':fallback_rejected',needs_review=True)


def consistent_with_primary(primary, fallback):
    """Reject new long spans and >500ms jumps at structurally stable Qwen anchors.

    This is a degradation gate, not evidence that Qwen anchors are true boundaries.
    """
    if not primary: return True
    if sum(w['end']-w['start']>1.5 for w in fallback)>sum(w['end']-w['start']>1.5 for w in primary):return False
    pairs=match_words(primary,fallback)['pairs']
    for qi,fi in pairs:
        q,f=primary[qi],fallback[fi]
        if .04<=q['end']-q['start']<=1.5 and max(abs(q['start']-f['start']),abs(q['end']-f['end']))>.5:return False
    return True

"""Word-anchored, bidirectional multi-event extraction. No inference of speaker count from channel count."""
import hashlib
import re


def add_entry_context(candidates, *, floor_checks=False):
 """Describe each onset and its immediate neighbors without changing word IDs."""
 by_speaker={}
 for c in candidates:
  by_speaker.setdefault(c['initiator'],[]).append(c)
 for own in by_speaker.values():
  for i,c in enumerate(own):
   previous=own[i-1] if i else None
   following=own[i+1] if i+1<len(own) else None
   other=by_speaker.get(c['recipient'],[])
   active=next((x for x in reversed(other) if x['start']<c['start']<x['end']),None)
   c['recipient_clause_at_onset']=(dict(text=active['text'],start=round(active['start'],4),
                                       end=round(active['end'],4)) if active else None)
   c['previous_own_candidate']=previous['candidate_id'] if previous else None
   c['previous_own_clause']=(dict(text=previous['text'],start=previous['start'],end=previous['end'])
                             if previous else None)
   c['next_own_candidate']=following['candidate_id'] if following else None
   c['linked_launch']=False
   if floor_checks:
    participant_after=[x['start'] for x in other if x['start']>=c['start']]
    c['recipient_resumes_after_onset']=round(min(participant_after)-c['start'],3) if participant_after else None
    c['own_pause_before']=round(c['start']-previous['end'],3) if previous else None
    clauses_before=[x for x in other if x['start']<c['start']]
    c['last_recipient_clause']=(dict(text=clauses_before[-1]['text'],
                                   end_minus_onset=round(clauses_before[-1]['end']-c['start'],3))
                               if clauses_before and active is None else None)
    answer=re.match(r"^(?:yes|no|yeah|yep|nope|i (?:do|did|am|have|can)|it (?:is|was))\b",c['text'],re.I)
    short_polarity=re.fullmatch(r'(?:yes|no|yeah|yep|nope|sure|absolutely)[.!?,]*',c['text'].strip(),re.I)
    c['terminal_question_answer']=bool(active and active['text'].rstrip().endswith(('?','？')) and
                                       0 < active['end']-c['start'] <= .5 and short_polarity and
                                       c['end']-c['start'] <= .6 and
                                       not active['alignment_flags'])
   if not following or not c['other_active_run_at_onset'] or c['end']-c['start']>.55:
    continue
   # A lone connective or subject can carry the start of the next clause.
   # Require a short gap, unfinished punctuation, and substantive follow-through.
   token=c['text'].strip().lower()
   if (re.fullmatch(r"(?:and|because|but|so|well|i|we|you|they|he|she|it|the|if)[,]?",token)
       and 0<=following['start']-c['end']<=.85 and len(following['word_ids'])>=3
       and not following['alignment_flags']):
    c['linked_launch']=True


def refine_entry_decisions(candidates, judgment):
 """Preserve separate feedback bursts and anchor a continuing launch at its first word."""
 decisions={d['candidate_id']:d for d in judgment['decisions']}
 by_id={c['candidate_id']:c for c in candidates}
 repairs=[]
 for c in candidates:
  d=decisions[c['candidate_id']]
  if c.get('linked_launch'):
   next_d=decisions[c['next_own_candidate']]
   if d['action']=='DROP' and next_d['action']=='KEEP' and next_d['label']=='INTERRUPT':
    repairs.append(dict(kind='earlier_launch',before=dict(d),continuation=dict(next_d)))
    d.update(action='KEEP',label='INTERRUPT',merge_into='',reason='First word of the same interruption.')
    next_d.update(action='MERGE',merge_into=d['candidate_id'])
  if d['action']=='MERGE' and d['label']=='BACKCHANNEL':
   parent=decisions.get(d['merge_into'])
   previous=by_id.get(c.get('previous_own_candidate'))
   if (parent and parent['action'] in ('KEEP','MERGE') and parent['label']=='BACKCHANNEL'
       and previous and c['start']-previous['end']>=.35 and c['other_active_run_at_onset']
       and c['overlap_seconds']>=.06 and not c['alignment_flags']
       and c['mix_feedback_supported']):
    repairs.append(dict(kind='separate_feedback_burst',before=dict(d)))
    d.update(action='KEEP',merge_into='',reason='Separate feedback burst with listener-audio support.')
 return repairs


def suppress_terminal_answers(candidates, judgment):
 """A polarity answer over the last instant of a clean question can be ordinary turn entry."""
 decisions={d['candidate_id']:d for d in judgment['decisions']}
 repairs=[]
 for c in candidates:
  d=decisions[c['candidate_id']]
  if c['terminal_question_answer'] and d['action']=='KEEP':
   repairs.append(dict(kind='terminal_question_answer',before=dict(d)))
   d.update(action='DROP',merge_into='',reason='Answer starts at the end of the question.')
 return repairs


def recover_echo_intervention(candidates, judgment):
 """Recover a repeated statement that overtakes an abandoned contrast and receives feedback."""
 decisions={d['candidate_id']:d for d in judgment['decisions']}
 by_speaker={}
 for c in candidates:
  by_speaker.setdefault(c['initiator'],[]).append(c)
 words=lambda text: re.findall(r"[a-z']+",text.lower())
 repairs=[]
 for c in candidates:
  d=decisions[c['candidate_id']]
  if d['action']!='DROP' or not c['other_active_run_at_onset'] or c['alignment_flags']:
   continue
  other=by_speaker[c['recipient']]
  for prior, fragment, response in zip(other,other[1:],other[2:]):
   if not (fragment['start']<c['start']<fragment['end']<response['start']<c['end']):
    continue
   preceding=words(prior['text'])
   repeated=words(c['text'])
   if (len(preceding)<3 or not prior['text'].rstrip().endswith('.') or
       words(fragment['text']) not in (['but'],['although']) or fragment['alignment_flags'] or
       len(repeated)<=len(preceding) or repeated[:len(preceding)]!=preceding or
       c['end']-fragment['end']<.55 or not response['mix_feedback_supported'] or response['alignment_flags'] or
       not response['other_active_run_at_onset'] or words(response['text']) not in
       (['yeah'],['yes'],['right'],['okay'],['uh','huh'],['mm','hmm'])):
    continue
   followup=decisions[response['candidate_id']]
   if followup['action'] not in ('DROP','KEEP') or (followup['action']=='KEEP' and followup['label']!='BACKCHANNEL'):
    continue
   repairs.append(dict(kind='echo_displaces_open_contrast',before=dict(d),response_before=dict(followup)))
   d.update(action='KEEP',label='INTERRUPT',merge_into='',reason='Echo displaces an unfinished contrast and keeps the floor.')
   followup.update(action='KEEP',label='BACKCHANNEL',merge_into='',reason='Independent acknowledgment during the intervening statement.')
   break
 return repairs

def suppress_supported_floor_continuation(candidates, judgment):
 """Require syntactic dependency, a failed short entry, and audio support together."""
 decisions={d['candidate_id']:d for d in judgment['decisions']}
 repairs=[]
 for c in candidates:
  d=decisions[c['candidate_id']]
  previous=c.get('previous_own_clause')
  parent=decisions.get(c.get('previous_own_candidate'))
  recipient=c.get('recipient_clause_at_onset')
  if not (d['action']=='KEEP' and d['label']=='INTERRUPT' and previous and parent and
          parent['action']=='DROP' and recipient and not c['alignment_flags']):continue
  open_tail=re.search(r'\b(?:your|my|our|their|the|a|an|and|or|of|with|for)\s*[,]*$',previous['text'],re.I)
  brief_failed_entry=len(recipient['text'].split())<=5 and not recipient['text'].rstrip().endswith(('.', '?','!'))
  if (open_tail and brief_failed_entry and c.get('audio_neither_probability',0)>=.95 and
      0<=c['start']-previous['end']<=1.5):
   repairs.append(dict(kind='audio_supported_open_clause_continuation',before=dict(d)))
   d.update(action='DROP',merge_into='',reason='Completes own open clause; competing fragment did not acquire floor.')
 return repairs
def intersections(a,b):
 out=[]
 for x in a:
  for y in b:
   lo=max(x['start'],y['start']);hi=min(x['end'],y['end'])
   if hi>lo:
    out.append([round(lo,4),round(hi,4)])
 merged=[]
 for lo,hi in sorted(out):
  if merged and lo<=merged[-1][1]:merged[-1][1]=max(hi,merged[-1][1])
  else:merged.append([lo,hi])
 return merged
def speaker_gate(case,paired=None):
 # Participant count is fixed by the dataset scope. Diarization clusters are
 # activity proposals and cannot change that count or block these cases.
 if len(case.get('streams',{}))!=2:
  return {'status':'UNVERIFIED','reason':'expected_two_participant_streams','source':'input_metadata'}
 if not paired or not paired.get('mixed_audio_checked'):
  return {'status':'UNVERIFIED','reason':'paired_audio_check_required','source':'paired_audio'}
 count=paired.get('mixed_cluster_count')
 return {'status':'PASS','reason':'paired_audio_available_for_known_two_participants','source':'paired_audio',
         'participant_count':2,'mixed_cluster_count':count,
         'speaker_attribution_anomaly':count is not None and count!=2}
def extract_candidates(case, *, refined=False, floor_checks=False):
 streams=case['streams'];units=[];runs={};word_meta={}
 for sp,words in streams.items():
  words=sorted(words,key=lambda w:(w['start'],w['end']))
  runs[sp]=[];prior=None
  for w in words:
   gap=None if prior is None else w['start']-prior['end']
   if not runs[sp] or gap>.5:runs[sp].append([w['start'],w['end']])
   else:runs[sp][-1][1]=max(runs[sp][-1][1],w['end'])
   word_meta[w['id']]={'speech_run_start':runs[sp][-1][0],'gap_from_previous_own_word':None if gap is None else round(gap,4)};prior=w
  group=[]
  for w in words:
   # Generate fine lexical proposals, not semantic labels. Pause/punctuation alone never implies interrupt.
   if group and (w['start']-group[-1]['end']>.35 or group[-1]['text'].rstrip().endswith(('.', '?','!', '。', '？', '！'))):
    units.append({'speaker':sp,'words':group});group=[]
   group.append(w)
  if group:units.append({'speaker':sp,'words':group})
 units.sort(key=lambda u:(u['words'][0]['start'],u['speaker']))
 result=[]
 for u in units:
  sp=u['speaker'];ws=u['words'];a,b=ws[0]['start'],max(w['end'] for w in ws)
  if ws[0].get('lexical_unit_start',ws[0]['id']) != ws[0]['id']:
   continue
  if not(case['interval'][0]<=a<case['interval'][1]):continue
  for other,ow in streams.items():
   if other==sp:continue
   overlap=intersections(ws,ow)
   # Broad proposal for recent floor ownership; later semantic judge must reject normal turn entry.
   recent=[w for w in ow if w['start']<a and w['end']>a-.7]
   uid=case['conversation_id']+':'+ws[0]['id']+'->'+other
   active=any(lo<a-.0001 and hi>a+.0001 for lo,hi in runs[other])
   result.append({'candidate_id':uid,'initiator':sp,'recipient':other,'start':a,'end':b,'word_ids':[w['id'] for w in ws],'text':' '.join(w['text'] for w in ws),'overlap_intervals':overlap,'overlap_seconds':round(sum(y-x for x,y in overlap),4),'other_recently_active':bool(recent),'other_active_run_at_onset':active,'proposal_kind':'ENTRY_OR_CONTINUATION' if active else 'CONTINUATION_ONLY',**word_meta[ws[0]['id']],'alignment_flags':['word_duration_over_2s'] if any(w['end']-w['start']>2 for w in ws) else []})
 if refined or floor_checks:
  add_entry_context(result, floor_checks=floor_checks)
 return result
def finalize(case,candidates,judgment,gate):
 byid={c['candidate_id']:c for c in candidates}; decisions=judgment['decisions']
 ids=[d['candidate_id'] for d in decisions]
 if len(ids)!=len(set(ids)) or set(ids)!=set(byid):raise ValueError('coverage_mismatch')
 result=[];lookup={};filtered=[];dropped=set()
 for d in sorted(decisions,key=lambda d:byid[d['candidate_id']]['start']):
  c=byid[d['candidate_id']]
  if d['action']=='DROP':dropped.add(c['candidate_id']);continue
  if d['label'] not in ('BACKCHANNEL','INTERRUPT'):raise ValueError('invalid_label')
  if d['action']=='MERGE':
   source=byid.get(d['merge_into'])
   if source is None or source['start']>=c['start']:raise ValueError('merge_parent_not_earlier')
   if (source['initiator'],source['recipient'])!=(c['initiator'],c['recipient']):raise ValueError('cross_direction_merge')
   if d['merge_into'] in dropped:
    dropped.add(c['candidate_id']);filtered.append({'candidate_id':c['candidate_id'],'reason':'continuation_of_dropped_non_intervention','model_decision':d});continue
   parent=lookup.get(d['merge_into'])
   if parent is None and any(x['candidate_id']==d['merge_into'] for x in filtered):
    filtered.append({'candidate_id':c['candidate_id'],'reason':'continuation_of_filtered_non_intervention','model_decision':d});continue
   if parent is None:raise ValueError('merge_parent_missing')
   if (parent['initiator'],parent['recipient'],parent['label'])!=(c['initiator'],c['recipient'],d['label']):raise ValueError('cross_direction_or_function_merge')
   parent['end']=max(parent['end'],c['end']);parent['candidate_ids'].append(c['candidate_id']);parent['texts'].append(c['text']);parent['overlap_intervals']=intersections([{'start':a,'end':b} for a,b in parent['overlap_intervals']+c['overlap_intervals']],[{'start':-1e9,'end':1e9}]);lookup[c['candidate_id']]=parent
  elif d['action']=='KEEP':
   if not c.get('other_active_run_at_onset',c['other_recently_active']):
    filtered.append({'candidate_id':c['candidate_id'],'reason':'future_overlap_only_no_recent_recipient_activity','model_decision':d});continue
   e={k:c[k] for k in ['initiator','recipient','start','end','overlap_intervals']};e.update(event_id=hashlib.sha256((c['candidate_id']+d['label']).encode()).hexdigest()[:20],label=d['label'],candidate_ids=[c['candidate_id']],texts=[c['text']],reason=d['reason'],clip_id=case['id'],in_original_target=case['target'][0]-0.0001<=c['start']<case['target'][1]-0.0001);result.append(e);lookup[c['candidate_id']]=e
  else:raise ValueError('invalid_action')
 for e in result:
  e['overlap_seconds']=round(sum(b-a for a,b in e['overlap_intervals']),4)
  e['visible_interval']=[max(e['start'],case['interval'][0]),min(e['end'],case['interval'][1])];e['extends_beyond_clip']=e['end']>case['interval'][1];e['alignment_flags']=sorted({f for cid in e['candidate_ids'] for f in byid[cid].get('alignment_flags',[])})
 return {'clip_id':case['id'],'speaker_gate':gate,'events':result,'filtered_candidates':filtered,'training_events':result if gate['status']=='PASS' else [],'annotation_source':'fresh_qwen_text_multi_event_v1','audio_semantic_judge_used':False}

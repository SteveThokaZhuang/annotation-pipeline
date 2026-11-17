import argparse,json,time,urllib.request,hashlib
from pathlib import Path
from event_extractor import (extract_candidates,finalize,recover_echo_intervention,refine_entry_decisions,
                             speaker_gate,suppress_terminal_answers,suppress_supported_floor_continuation)
D=Path(__file__).resolve().parent
RULES=(D/'interaction_multi_event_system.txt').read_text(encoding='utf8')
schema={'type':'object','properties':{'decisions':{'type':'array','items':{'type':'object','properties':{'candidate_id':{'type':'string'},'action':{'type':'string','enum':['KEEP','MERGE','DROP']},'label':{'type':'string','enum':['BACKCHANNEL','INTERRUPT']},'merge_into':{'type':'string'},'reason':{'type':'string'}},'required':['candidate_id','action','label','merge_into','reason'],'additionalProperties':False}}},'required':['decisions'],'additionalProperties':False}
def classify(case, audio=None, diarization=None, audio_view='mix', concise=False, recover_invalid_merge=False,
             refined=False, floor_checks=False, echo_recovery=True, resume_truncated=False,
             endpoint='http://127.0.0.1:18770/v1/chat/completions', model='qwen38-27b'):
 candidates=extract_candidates(case, refined=refined, floor_checks=floor_checks)
 if audio is not None:
  if audio.get('clip_id')!=case['id']:
   raise ValueError('mixed_audio_clip_id_mismatch')
  votes=audio.get('events',[])
  vote_ids=[e['candidate_id'] for e in votes]
  required={c['candidate_id'] for c in candidates if c['other_active_run_at_onset'] and c['overlap_seconds']>0}
  known={c['candidate_id'] for c in candidates}
  if (not votes or len(set(vote_ids))!=len(vote_ids) or not required<=set(vote_ids)<=known or
      any(e[audio_view]['answer'] not in ('INTERRUPT','BACKCHANNEL','NEITHER') for e in votes)):
   raise ValueError('mixed_audio_votes_missing_or_invalid')
 if refined:
  audio_labels={e['candidate_id']:e[audio_view]['answer'] for e in audio['events']} if audio else {}
  for c in candidates:
   c['mix_feedback_supported']=audio_labels.get(c['candidate_id'])=='BACKCHANNEL'
   vote=next((e[audio_view] for e in audio['events'] if e['candidate_id']==c['candidate_id']),{}) if audio else {}
   c['audio_neither_probability']=vote.get('probs',{}).get('NEITHER',0)
 payload={'analysis_interval':case['interval'],'speakers':list(case['streams']),'context':case['context'],'candidates':[{k:v for k,v in c.items() if k not in ['word_ids','overlap_intervals']} for c in candidates]}
 mixed=diarization.get('mix') if diarization else None
 gate=speaker_gate(case,dict(mixed_audio_checked=True,
                            mixed_cluster_count=mixed.get('count') if mixed is not None else None)
                   if audio is not None else None)
 if audio:
  if audio['clip_id']!=case['id']:
   raise ValueError('mixed_audio_clip_id_mismatch')
  payload['audio_candidate_votes']=[dict(candidate_id=e['candidate_id'],label=e[audio_view]['answer'])
                                    for e in audio['events']]
 rules=RULES
 if refined:
  rules+='\n'+(D/'interaction_entry_refinement.txt').read_text(encoding='utf8')
  for item in payload['candidates']:
   item.pop('mix_feedback_supported')
   item.pop('audio_neither_probability',None)
   item.pop('previous_own_clause',None)
 if floor_checks:
  rules+='\n'+(D/'interaction_floor_checks.txt').read_text(encoding='utf8')
 aliases={}
 if concise:
  rules+=' Keep each reason to at most 8 words. A short reason is enough for DROP.'
  aliases={c['candidate_id']:f"C{i}" for i,c in enumerate(candidates,1)}
  for item in payload['candidates']+payload.get('audio_candidate_votes',[]):
   item['candidate_id']=aliases[item['candidate_id']]
   for key in ('previous_own_candidate','next_own_candidate'):
    if item.get(key):
     item[key]=aliases[item[key]]
  reverse_aliases={short:original for original,short in aliases.items()}
 attempts=[]
 keyed=bool(resume_truncated and aliases)
 for attempt in range(3):
  output_rules,output_schema,output_payload=rules,schema,payload
  if keyed:
   props={}
   for c in candidates:
    parents=[aliases[p['candidate_id']] for p in candidates if p['start']<c['start'] and
             (p['initiator'],p['recipient'])==(c['initiator'],c['recipient'])]
    fields=dict(schema['properties']['decisions']['items']['properties'])
    fields.pop('candidate_id')
    fields['action']={'type':'string','enum':['KEEP','DROP','MERGE'] if parents else ['KEEP','DROP']}
    fields['merge_into']={'type':'string','enum':['',*parents]}
    fields['reason']={'type':'string','maxLength':120}
    props[aliases[c['candidate_id']]]={'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
   output_schema={'type':'object','properties':props,'required':list(props),'additionalProperties':False}
   output_rules+=('\nOutput format override: emit one top-level JSON object keyed by exact candidate alias. '
                  'Each value contains action, label, merge_into, and a brief reason. '
                  'Use an empty merge_into unless the action is MERGE. Do not emit a decisions array. '
                  'The input candidates are rows with fields in the order given by candidate_columns.')
   columns=list(dict.fromkeys(k for c in payload['candidates'] for k in c))
   output_payload={**payload,'candidate_columns':columns,
                   'candidates':[[c.get(k) for k in columns] for c in payload['candidates']]}
  payload_text=json.dumps(output_payload,separators=(',',':')) if keyed else json.dumps(output_payload)
  body={'model':model,'messages':[{'role':'system','content':output_rules},{'role':'user','content':payload_text}],'temperature':0,'max_tokens':3072,'chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_schema','json_schema':{'name':'directed_multi_events','strict':True,'schema':output_schema}}}
  if attempts:body['messages'].append({'role':'user','content':'Previous output failed structural validation: '+attempts[-1].get('error','')+'. Return full corrected decisions; preserve ALL candidate IDs exactly.'})
  attempt_log={'request_sent':False}
  attempts.append(attempt_log)
  start=time.monotonic()
  try:
   request=urllib.request.Request(endpoint,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
   attempt_log['request_sent']=True
   with urllib.request.urlopen(request,timeout=300) as f:response=json.load(f)
   choice=response['choices'][0];attempt_log.update({'response':response,'seconds':time.monotonic()-start,'request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),'keyed_output':keyed,'finish_reason':choice['finish_reason'],'usage':response.get('usage')})
   if choice['finish_reason']=='length':
    keyed=bool(aliases)
    raise ValueError('generation_truncated')
   if choice['finish_reason']!='stop':
    raise ValueError('unexpected_finish_reason:'+str(choice['finish_reason']))
   judgment=json.loads(choice['message']['content'])
   if keyed:
    judgment={'decisions':[dict(judgment[key],candidate_id=key) for key in reverse_aliases]}
   if aliases:
    for decision in judgment['decisions']:
     decision['candidate_id']=reverse_aliases[decision['candidate_id']]
     if decision['merge_into']:
      decision['merge_into']=reverse_aliases[decision['merge_into']]
   repairs=[]
   if keyed:
    # A failed continuation link cannot add an independent event.
    by_id={c['candidate_id']:c for c in candidates}
    for d in judgment['decisions']:
     if d['action']=='MERGE':
      c=by_id[d['candidate_id']]
      parent=by_id.get(d['merge_into'])
      if (parent is None or parent['start']>=c['start'] or
          (parent['initiator'],parent['recipient'])!=(c['initiator'],c['recipient'])):
       repairs.append(dict(d))
       d.update(action='DROP',merge_into='',reason='Invalid continuation: no earlier same-direction parent.')
   entry_repairs=refine_entry_decisions(candidates,judgment) if refined else []
   if floor_checks:
    entry_repairs.extend(suppress_terminal_answers(candidates,judgment))
    if echo_recovery:
     entry_repairs.extend(recover_echo_intervention(candidates,judgment))
    entry_repairs.extend(suppress_supported_floor_continuation(candidates,judgment))
   try:
    result=finalize(case,candidates,judgment,gate)
   except ValueError as exc:
    if not recover_invalid_merge or str(exc)!='cross_direction_or_function_merge':
     raise
    # The candidate was not asserted to initiate an event. Drop a conflicting
    # continuation without rewriting the labels or onsets of the kept events.
    decisions={d['candidate_id']:d for d in judgment['decisions']}
    for candidate in candidates:
     decision=decisions[candidate['candidate_id']]
     if decision['action']=='MERGE' and decision['label']!=decisions[decision['merge_into']]['label']:
      repairs.append(dict(decision))
      decision.update(action='DROP',merge_into='',reason='Invalid continuation: function differs from parent.')
    if not repairs:
     raise
    result=finalize(case,candidates,judgment,gate)
   result.update(number=case['number'],old_label=case['original_label'],target=case['target'],candidates=candidates,judgment=judgment,attempts=attempts,status='ok',invalid_merges_removed=repairs)
   if refined:
    result['entry_repairs']=entry_repairs
   if audio:
    result.update(annotation_source='qwen_text_plus_jev_audio',audio_semantic_judge_used=True,
                  audio_view=audio_view)
   break
  except Exception as e:
   attempt_log.update(error=str(e),seconds=time.monotonic()-start)
   if getattr(e,'code',None)==400 and aliases:
    keyed=True
 else:result={'clip_id':case['id'],'number':case['number'],'status':'error','attempts':attempts}
 return result

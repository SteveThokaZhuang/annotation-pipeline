"""Sequential fresh-process AMI-100 benchmark on exactly one H100.

No model concurrency, no prediction reuse, no policy fitting. End-to-end wall
clock RTF is the comparable primary speed metric for all four implementations.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time

from common import ROOT, DATASET, load_manifest, read_jsonl, write_jsonl
from evaluate import word_rows, summarize

BACKENDS=('qwen','mfa','nyra','hybrid')
INVALID={'zero_duration','negative_duration','nonfinite','out_of_bounds','nonmonotonic','overlap'}

def accuracy(items):
    """Missing or structurally invalid predictions fail the all-reference denominator."""
    n=len(items)
    result=summarize(items)
    for ms in (50,100,200):
        passed=sum(w['mae'] is not None and not (INVALID & set(w['categories']))
                   and max(w['start_error'],w['end_error'])<=ms/1000+1e-9 for w in items)
        result[f'valid_both_boundaries_acc_{ms}ms']=passed/n if n else None
        result[f'valid_both_boundaries_correct_{ms}ms']=passed
    result['legacy_mae_le100ms_accuracy']=(n-result['bad_or_missing'])/n if n else None
    return result

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def query_gpus():
    s=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version','--format=csv,noheader,nounits'],text=True)
    return [dict(zip(('index','uuid','name','memory_mib','driver'),[v.strip() for v in row])) for row in csv.reader(s.splitlines())]

class Monitor:
    def __init__(self):
        self.samples=[];self.active='idle';self.stop=threading.Event();self.thread=threading.Thread(target=self.loop,daemon=True)
    def loop(self):
        while not self.stop.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,stderr=subprocess.STDOUT,timeout=8)
                apps=[dict(zip(('gpu_uuid','pid','used_gpu_memory_mib'),[v.strip() for v in row])) for row in csv.reader(raw.splitlines()) if row]
                self.samples.append(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),active=self.active,applications=apps))
            except Exception as exc:self.samples.append(dict(active=self.active,error=str(exc)))
            self.stop.wait(1)
    def start(self):self.thread.start()
    def close(self):self.stop.set();self.thread.join(timeout=10)


def render_report(output,records,metadata,monitor):
    summary={}
    for backend in BACKENDS:
        rr=[r for r in records if r['backend']==backend]
        seconds=[r['wall_seconds'] for r in rr]
        summary[backend]=dict(trials=len(rr),median_seconds=statistics.median(seconds),min_seconds=min(seconds),max_seconds=max(seconds),
            median_rtf=statistics.median(seconds)/metadata['audio_seconds'],
            min_rtf=min(seconds)/metadata['audio_seconds'],max_rtf=max(seconds)/metadata['audio_seconds'],
            accuracy_first_trial=rr[0]['accuracy'],
            filtered_mae_ms_range=[min(r['accuracy']['filtered']['mae_ms'] for r in rr),max(r['accuracy']['filtered']['mae_ms'] for r in rr)],
            filtered_acc100_range=[min(r['accuracy']['filtered']['valid_both_boundaries_acc_100ms'] for r in rr),max(r['accuracy']['filtered']['valid_both_boundaries_acc_100ms'] for r in rr)],
            semantic_prediction_hashes=[r['semantic_prediction_sha256'] for r in rr])
    report=dict(metadata=metadata,systems=summary,records=records,observed_compute_gpu_uuids=sorted({a['gpu_uuid'] for s in monitor for a in s.get('applications',[])}))
    (output/'comparison.json').write_text(json.dumps(report,indent=2))
    doc=ROOT/'docs/fa/07_single_h100_v2_benchmark.md'
    rel=output.relative_to(ROOT)
    lines=['# V2 单张 H100：AMI-100 准确度与 RTF','','## 本次设置','',
        f"时间：{metadata['started_utc']}；100条，{metadata['audio_seconds']:.2f}秒，758参考词；过滤参考468词。只对齐 manifest 原文本，不在本次重新跑ASR。所有方案使用相同音频、文本与词匹配方法。",'',
        f"固定GPU：物理{metadata['gpu']['index']}，`{metadata['gpu']['uuid']}`，{metadata['gpu']['name']}，{metadata['gpu']['memory_mib']} MiB。所有GPU子进程 `CUDA_VISIBLE_DEVICES` 固定为该UUID，逻辑设备均为cuda:0；worker强制检查可见GPU数为1。MFA使用CPU，明确设置CUDA不可见，不占用额外GPU。",'',
        f"冻结V2策略SHA256：`{metadata['policy_sha256']}`。按先后顺序：填充词比例≥5%走MFA；否则Qwen最大词间隙≥0.5秒走Nyra；否则保留Qwen。原V2防退化门不变。没有拟合新参数、没有挑最优结果、没有利用金标准选路由。",'',
        'Qwen BF16+SDPA、逐条处理；Nyra FP32 WavLM、逐条处理并保留um/uh约定映射；torch线程4。MFA english_mfa/english_us_mfa 3.1.0，beam400/retry4000、8 jobs、每job BLAS/OMP线程1，逐段独立speaker目录、禁用说话人自适应。混合使用同样配置。', '',
        f"每方案{metadata['repeats']}次，每次新进程、新输出目录；轮换顺序见metadata.json，严格串行，前一个进程退出后才开始下一个。没有复用预测。模型权重已在本地；OS/文件系统/Numba缓存未清空，不能称为首次安装或磁盘冷缓存测试。GPU活动以1秒间隔监控，原始记录保留。",'',
        '## 准确度定义','',
        '**Acc@100ms**：参考词与预测词完全词形匹配、预测时间结构合法（正时长、有限、在片段内、无超过容差的逆序/重叠），并且起点、终点各自的绝对误差均≤100ms。分母为全部相应参考词；未匹配和零时长计失败。另保存Acc@50/200ms。浮点阈值容差1e-9秒。', '',
        'MAE仍为每词起止误差均值，再对匹配词求均值。另保留旧口径“词MAE≤100ms比例”，它可能把零时长/单边界偏差较大的词算入，不能与本页主准确率混用。', '',
        'AMI逐词边界仍是自动对齐银标准，不是独立人工金标准。完整口径758词；过滤口径468词，排除参考可疑时长、截断、重叠及非法边界；过滤只影响评分，不删除推理输入。这里不是ASR词识别正确率，也不是盲测泛化评价。','',
        '下表准确度取第1轮；三轮输出是否一致及数值范围另列。RTF使用全部轮次的中位数。','',
        '| 方案 | 完整Acc@100ms | 过滤Acc@100ms | 完整MAE ms | 过滤MAE ms | 零时长词 | 端到端秒 中位数 | 端到端RTF 中位数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for b in BACKENDS:
        s=summary[b];a=s['accuracy_first_trial'];f=a['filtered'];full=a['full']
        lines.append(f"| {b} | {full['valid_both_boundaries_acc_100ms']:.2%} | {f['valid_both_boundaries_acc_100ms']:.2%} | {full['mae_ms']:.2f} | {f['mae_ms']:.2f} | {full['categories'].get('zero_duration',0)} | {s['median_seconds']:.3f} | {s['median_rtf']:.5f} |")
    lines+=['','## RTF计时边界','',
        '**端到端RTF = 从启动该方案进程到其退出的实测墙钟秒数 / 466.36秒音频**。包括解释器/库导入、模型加载、音频读取、对齐、路由、回退、MFA输入整理及TextGrid导出解析、JSON输出和进程退出。准确度评估、GPU监控文件保存、模型下载不计入。RTF越低越快，<1表示快于实时。', '',
        '混合速度由实际运行 `hybrid.py` 测得：Qwen全量，然后只对命中子集启动Nyra/MFA。被验收拒绝的候选也消耗时间，计入RTF；不能用全量模型各自RTF加权来替代。当前是子进程批处理实现，有模型/解释器重复启动成本，不能解释为常驻服务的理想吞吐。','',
        '| 方案 | 每轮端到端秒 | 每轮RTF | 过滤MAE范围 ms | 过滤Acc@100ms范围 |',
        '|---|---|---|---|---|']
    for b in BACKENDS:
        rr=[r for r in records if r['backend']==b];s=summary[b]
        time_cells=', '.join(format(r['wall_seconds'],'.3f') for r in rr)
        rtf_cells=', '.join(format(r['rtf'],'.5f') for r in rr)
        lines.append(f"| {b} | {time_cells} | {rtf_cells} | {s['filtered_mae_ms_range'][0]:.3f}–{s['filtered_mae_ms_range'][1]:.3f} | {s['filtered_acc100_range'][0]:.2%}–{s['filtered_acc100_range'][1]:.2%} |")
    lines+=['','## 阶段计时（辅助口径）','',
        'Qwen/Nyra的processing_seconds从模型完整加载并同步GPU后开始，包含逐条音频读取、预处理、对齐、同步及JSON输出，含首个推理调用，**没有预热，不能称为稳定热推理**。Nyra的惰性WavLM权重加载已移至模型加载阶段。MFA没有可对等分离的加载后热推理阶段，其alignment_subprocess_seconds仍包含整个新CLI启动/加载/特征/对齐/导出。因此此表用于解释开销；横向速度排名以端到端RTF为准。','',
        '| 方案 | 轮次 | 加载阶段秒 | 加载后处理秒 | 对应处理RTF | MFA完整CLI秒 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in records:
        b=r['backend'];ph=r['phases']
        if b in ('qwen','nyra'):
            lines.append(f"| {b} | {r['repeat']+1} | {ph['model_load_seconds']:.3f} | {ph['processing_seconds']:.3f} | {ph['processing_seconds']/metadata['audio_seconds']:.5f} | — |")
        elif b=='mfa':lines.append(f"| {b} | {r['repeat']+1} | 无法单独分离 | — | — | {ph['alignment_subprocess_seconds']:.3f} |")
    lines+=['','## 混合实际调用量','']
    for r in records:
        if r['backend']=='hybrid':
            ph=r['phases'];lines.append(f"- 第{r['repeat']+1}轮：请求{ph['requested']}；实际选择{ph['selected']}；需复核{ph['needs_review']}段。各子进程墙钟时间：{[round(s['seconds'],3) for s in ph['stages']]}秒（Qwen→Nyra→MFA）。")
    q=summary['qwen'];h=summary['hybrid'];f=h['accuracy_first_trial']['filtered'];fq=q['accuracy_first_trial']['filtered']
    lines+=['','## 结论','',
        f"本次V2相对Qwen：过滤MAE变化{fq['mae_ms']:.2f}→{f['mae_ms']:.2f}ms，主准确率{fq['valid_both_boundaries_acc_100ms']:.2%}→{f['valid_both_boundaries_acc_100ms']:.2%}；端到端耗时为Qwen的{h['median_seconds']/q['median_seconds']:.2f}倍。准确度提升与处理开销应同时考虑，不能把质量改进说成速度提升。",'',
        '零时长消失不等于对齐正确：MFA/Nyra仍可能出现较大偏移，完整与过滤MAE必须同时看。本页结论只适用于这一固定AMI-100、这些模型配置及单张H100下的当前实现；不把共享文件系统的IO等待或冷进程开销解释为GPU本身计算速度。','',
        f"原始结果：`{rel}`。`comparison.json`汇总全部轮次，`trials.jsonl`逐次记录墙钟时间/命令/准确度/阶段数据，`gpu-monitor.jsonl`记录GPU活动，`metadata.json`记录硬件/环境/策略和代码摘要。每轮保留全量预测、逐词误差和日志；不会覆盖此前实验。",'',
        '复现：','', '```bash',f'python src/v1/fa/benchmark_single_h100.py --output {rel}-rerun --gpu 0 --repeats 3','```']
    doc.write_text('\n'.join(lines)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=DATASET/'manifest.jsonl')
    p.add_argument('--policy',type=Path,default=ROOT/'results/fa/policy.json')
    p.add_argument('--gpu',default='0');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--qwen-python',default='/mnt/data/L202500293/obed/envs/qwenasr/bin/python')
    p.add_argument('--mfa-python',default=str(ROOT/'models/fa-runtime/mfa/bin/python'))
    a=p.parse_args();output=a.output.resolve()
    if a.repeats<1:p.error('repeats must be positive')
    if output.exists():raise FileExistsError('Use a fresh output directory; prediction reuse is forbidden')
    gpu=next(g for g in query_gpus() if g['index']==a.gpu or g['uuid']==a.gpu)
    if 'H100' not in gpu['name']:raise RuntimeError('An H100 is required')
    policy=json.loads(a.policy.read_text());assert policy['version']==2 and policy['guarded']
    rows=load_manifest(a.manifest)
    import soundfile as sf
    for r in rows:
        info=sf.info(r['audio'])
        assert info.samplerate==16000 and info.channels==1
        assert abs(info.duration-r['duration'])<=1/16000+1e-6
    seconds=sum(r['duration'] for r in rows)
    output.mkdir(parents=True)
    (output/'policy.json').write_bytes(a.policy.read_bytes())
    manifest=output/'inference_manifest.jsonl'
    write_jsonl(manifest,[{k:r[k] for k in ('id','audio','text','duration','speaker','language') if k in r} for r in rows])
    orders=[list(BACKENDS[i%4:]+BACKENDS[:i%4]) for i in range(a.repeats)]
    meta=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=gpu,repeats=a.repeats,orders=orders,
        samples=len(rows),audio_seconds=seconds,reference_words=sum(len(r['words']) for r in rows),
        source_manifest=str(a.manifest),source_manifest_sha256=sha(a.manifest),policy_sha256=sha(a.policy),
        qwen_python=a.qwen_python,mfa_python=a.mfa_python,host=platform.node(),platform=platform.platform(),
        cpu_count=os.cpu_count(),load_average_start=os.getloadavg(),
        source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        checkpoint_provenance={name:json.loads(path.read_text()) for name,path in [('qwen',ROOT/'models/qwen3-forced-aligner/checkpoints/download-provenance.json'),('nyra',ROOT/'models/nyra-forced-aligner/checkpoints/download-provenance.json')]},
        cpu_threads=dict(torch=4,mfa_jobs=8,mfa_blas_threads_per_job=1),caches='OS/filesystem/Numba caches retained; fresh processes and predictions each trial',
        accuracy_definition='valid positive duration and both boundaries within threshold; missing/invalid fail; all reference words denominator',
        timing_scope='wall clock from Popen launch through exit; full actual execution; evaluation excluded')
    (output/'metadata.json').write_text(json.dumps(meta,indent=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],FA_REQUIRE_SINGLE_GPU='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONHASHSEED='0')
    env['PYTHONPATH']=str(ROOT/'models/fa-runtime/nyra')+os.pathsep+env.get('PYTHONPATH','')
    here=Path(__file__).parent;records=[];monitor=Monitor();monitor.start()
    try:
        for repeat,order in enumerate(orders):
            for b in order:
                trial=output/f'repeat{repeat+1}'/b;trial.mkdir(parents=True)
                if b in ('qwen','nyra'):
                    cmd=[a.qwen_python,str(here/'run_aligners.py'),'--backend',b,'--manifest',str(manifest),'--output',str(trial/'predictions.jsonl'),'--device','cuda:0']
                    predictions=trial/'predictions.jsonl';runtime=trial/'predictions.runtime.json'
                elif b=='mfa':
                    cmd=[a.mfa_python,str(here/'run_mfa.py'),'--manifest',str(manifest),'--output',str(trial)]
                    predictions=trial/'predictions.jsonl';runtime=trial/'runtime.json'
                else:
                    cmd=[sys.executable,str(here/'hybrid.py'),'--manifest',str(manifest),'--output',str(trial),'--policy',str(output/'policy.json'),'--qwen-python',a.qwen_python,'--mfa-python',a.mfa_python,'--device','cuda:0']
                    predictions=trial/'hybrid.jsonl';runtime=trial/'runtime.json'
                print(f'START round={repeat+1}/{a.repeats} backend={b} gpu={gpu["uuid"]}',flush=True)
                (output/'status.json').write_text(json.dumps(dict(state='running',repeat=repeat+1,backend=b,completed=len(records)),indent=2))
                monitor.active=f'repeat{repeat+1}/{b}'
                before=resource.getrusage(resource.RUSAGE_CHILDREN)
                start=time.perf_counter()
                with (trial/'process.log').open('w') as log:
                    result=subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                elapsed=time.perf_counter()-start
                after=resource.getrusage(resource.RUSAGE_CHILDREN);monitor.active='evaluation'
                if result.returncode:raise RuntimeError(f'{b} failed ({result.returncode}): {trial}/process.log')
                pp=read_jsonl(predictions);byid={v['id']:v for v in pp}
                assert len(pp)==len(rows) and set(byid)=={r['id'] for r in rows}
                assert not any(v.get('error') for v in pp),f'{b} contains execution errors'
                details=[dict(w,backend=b,repeat=repeat+1) for r in rows for w in word_rows(r,byid[r['id']])]
                write_jsonl(trial/'word_diagnostics.jsonl',details)
                phases=json.loads(runtime.read_text())
                gpu_workers=[]
                if b in ('qwen','nyra'):gpu_workers=[phases]
                elif b=='hybrid':
                    gpu_workers=[json.loads((trial/'qwen.runtime.json').read_text())]
                    if (trial/'nyra.runtime.json').exists():gpu_workers.append(json.loads((trial/'nyra.runtime.json').read_text()))
                    # Validate fallback execution even when the router retains primary output.
                    for name in ('nyra.jsonl','mfa/predictions.jsonl'):
                        path=trial/name
                        if path.exists():assert not any(v.get('error') for v in read_jsonl(path)),f'{b}: failed fallback'
                    assert all(s['returncode']==0 for s in phases['stages'])
                for info in gpu_workers:
                    assert info['visible_gpu_count']==1 and info['cuda_visible_devices']==gpu['uuid'] and 'H100' in info['gpu_name']
                    assert info['resumed_samples']==0
                semantic=[dict(id=v['id'],words=v['words'],selected_backend=v.get('selected_backend')) for v in pp]
                record=dict(backend=b,repeat=repeat,wall_seconds=elapsed,rtf=elapsed/seconds,
                    audio_seconds=seconds,samples=len(pp),returncode=result.returncode,command=cmd,
                    cpu_user_seconds=after.ru_utime-before.ru_utime,cpu_system_seconds=after.ru_stime-before.ru_stime,
                    accuracy=dict(full=accuracy(details),filtered=accuracy([w for w in details if w['filtered']])),
                    phases=phases,predictions=str(predictions),semantic_prediction_sha256=hashlib.sha256(json.dumps(semantic,sort_keys=True).encode()).hexdigest())
                records.append(record);write_jsonl(output/'trials.jsonl',records)
                print(f'DONE round={repeat+1} backend={b} wall={elapsed:.3f}s RTF={record["rtf"]:.5f} filtered_acc100={record["accuracy"]["filtered"]["valid_both_boundaries_acc_100ms"]:.4%}',flush=True)
        assert sha(a.policy)==meta['policy_sha256']
    finally:
        monitor.close();write_jsonl(output/'gpu-monitor.jsonl',monitor.samples)
    report=render_report(output,records,meta,monitor.samples)
    (output/'status.json').write_text(json.dumps(dict(state='complete',completed=len(records)),indent=2))
    print(json.dumps({b:{k:v for k,v in s.items() if k in ('median_seconds','median_rtf','filtered_mae_ms_range','filtered_acc100_range')} for b,s in report['systems'].items()},indent=2))

if __name__=='__main__':main()

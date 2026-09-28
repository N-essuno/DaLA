"""Scale parser and checker pools on measured queue/CPU evidence.

Resizes preserve immutable checkpoints, generation lineage and language inputs.
No checker failures are suppressed. Each resize has a stop/migration/launch receipt.
"""
import argparse,copy,json,math,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from dala.batch_pipeline import atomic_json
from dala.common_pile import sha256_file
from scripts.report_european_scale_status import snapshot

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/tmp/dala-six-venv/bin/python'


def decision(workers,instances,ahead,checker_cores,parser_cores,host_cores,capacity,available_gb):
    proposed=max(4,math.ceil(instances*1.5))
    extra=proposed-instances
    if ahead < .7*workers*4:return None,'parser_queue_not_full'
    if checker_cores < 3*instances:return None,'checker_cpu_not_saturated'
    if parser_cores >= .75*workers:return None,'parser_capacity_limited'
    if host_cores+extra*10 > .9*capacity:return None,'cpu_headroom'
    if available_gb < 16+extra*3:return None,'memory_headroom'
    return proposed,'checker_bottleneck'



def capacity_decision(workers,instances,ahead,checker_cores,parser_cores,host_cores,capacity,available_gb,parser_rss_gb=2):
    checkers,reason=decision(workers,instances,ahead,checker_cores,parser_cores,host_cores,capacity,available_gb)
    if reason=='parser_capacity_limited':
        checkers,reason=decision(workers,instances,ahead,checker_cores,0,host_cores,capacity,available_gb)
    if checkers:return {'kind':'checker','instances':checkers,'parser_workers':workers},reason
    # Low ready queue plus busy parsers means additional checkers cannot help.
    if ahead < .5*workers*4 and parser_cores >= .65*workers:
        proposed=math.ceil(workers*1.5);extra=proposed-workers
        if host_cores+extra > .9*capacity:return None,'cpu_headroom'
        if available_gb < 16+extra*max(1,1.3*parser_rss_gb):return None,'memory_headroom'
        return {'kind':'parser','instances':instances,'parser_workers':proposed},'parser_bottleneck'
    return None,reason


def process_sample(parents,seconds=5):
    import psutil
    groups={}
    for lang,pid in parents.items():
        try:procs=[psutil.Process(pid),*psutil.Process(pid).children(recursive=True)]
        except psutil.Error:continue
        group=[]
        for p in procs:
            try:
                cmd=' '.join(p.cmdline());kind='checker' if 'java' in cmd else 'parser' if 'multiprocessing.spawn' in cmd else 'other'
                t=p.cpu_times();group.append((p,kind,t.user+t.system))
            except psutil.Error:pass
        groups[lang]=group
    host=psutil.cpu_percent(interval=seconds)*len(os.sched_getaffinity(0))/100
    results={}
    for lang,group in groups.items():
        totals={};parser_rss=0
        for p,kind,before in group:
            try:
                t=p.cpu_times();totals[kind]=totals.get(kind,0)+(t.user+t.system-before)/seconds
                if kind=='parser':parser_rss=max(parser_rss,p.memory_info().rss/2**30)
            except psutil.Error:pass
        totals['_parser_rss_gb']=parser_rss
        results[lang]=totals
    return results,host


def resize(launch_path,language,parent_pid,instances,evidence,receipt_root,parser_workers=None):
    import psutil
    launch=json.loads(launch_path.read_text());record=next(r for r in launch['languages'] if r['language']==language)
    process=psutil.Process(parent_pid)
    if process.ppid()!=record.get('supervisor_pid',launch['pid']) or f'--languages {language} ' not in ' '.join(process.cmdline())+' ':
        raise ValueError('Worker identity changed; refusing stop')
    profile_path=Path(record['profile']);old=json.loads(profile_path.read_text())
    if sha256_file(profile_path)!=record['profile_sha256']:raise ValueError('Profile changed')
    code_dir=Path(record['cwd'])/'dala'
    current={str(p.relative_to(code_dir)):sha256_file(p) for p in code_dir.rglob('*.py')}
    if current!=record['code_sha256']:raise ValueError('Generator changed')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');run_id=f'checker_auto_{stamp}_{language}'
    folder=receipt_root/run_id;folder.mkdir()
    checkpoint=ROOT/'la_output/european_pilots'/f'{run_id}_checkpoints'/language;checkpoint.parent.mkdir()
    new=copy.deepcopy(old);new['build']['checkpoint_dir']=str(checkpoint);new['checker'].update(instances=instances,workers=instances*8)
    if parser_workers is not None:new['build']['parser_processes']=parser_workers
    newprofile=ROOT/'la_output/resources/european-expansion'/run_id/language/'profile.json';newprofile.parent.mkdir(parents=True);atomic_json(newprofile,new)
    oldprogress=Path(old['build']['checkpoint_dir'])/'progress.json'
    receipt=dict(language=language,at=datetime.now(timezone.utc).isoformat(),old_profile=str(profile_path),new_profile=str(newprofile),evidence=evidence)
    atomic_json(folder/'resize.json',receipt)
    procs=[process,*process.children(recursive=True)]
    for p in procs:
        try:p.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(procs,timeout=10)
    if any(p.is_running() and p.status()!=psutil.STATUS_ZOMBIE for p in alive):raise RuntimeError('Owned processes did not exit')
    progress=json.loads(oldprogress.read_text());receipt['preserved_progress']=progress;atomic_json(folder/'resize.json',receipt)
    command=[PYTHON,'-u','-m','scripts.migrate_execution_profile',str(profile_path),str(newprofile)]
    with (folder/'migration.log').open('w') as f:subprocess.run(command,cwd=record['cwd'],stdout=f,stderr=subprocess.STDOUT,check=True)
    command=[PYTHON,'-u','-m','scripts.run_european_pilots','--languages',language,'--workers','1','--run-id',run_id,'--profile-root',str(newprofile.parent.parent),'--max-documents','0']
    with (folder/'supervisor.log').open('w') as f:newproc=subprocess.Popen(command,cwd=record['cwd'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    record.update(parser_workers=new['build']['parser_processes'],profile=str(newprofile),profile_sha256=sha256_file(newprofile),checkpoint_dir=str(checkpoint),output=str(ROOT/'la_output/european_pilots'/run_id/language),run_id=run_id,supervisor_pid=newproc.pid,external=True,language_logs=str(ROOT/'wiki/artifacts/european-pilots'/run_id),resume_progress=progress)
    launch['parser_workers_total']=sum(r['parser_workers'] for r in launch['languages'])
    atomic_json(launch_path,launch)
    receipt.update(supervisor_pid=newproc.pid,command=command,status='launched');atomic_json(folder/'resize.json',receipt)
    return receipt


def main():
    import psutil
    p=argparse.ArgumentParser();p.add_argument('launch',type=Path);p.add_argument('--languages',nargs='+',default=['fr','uk','el']);p.add_argument('--interval',type=int,default=60);args=p.parse_args()
    output=ROOT/'wiki/artifacts/european-expansion/checker-autoscaling';output.mkdir(exist_ok=True)
    baselines={};last_resize={};previous_rates={};plateaus={}
    while True:
        s=snapshot(args.launch);launch=json.loads(args.launch.read_text());records={r['language']:r for r in launch['languages']}
        active={l:v['pid'] for l,v in s['languages'].items() if l in args.languages and v['pid']}
        if not active:
            atomic_json(output/'status.json',dict(status='finished_no_live_target_jobs',at=s['timestamp']));return
        cpus,host=process_sample(active);now=time.time();observations={};action=None
        for l,pid in active.items():
            r=records[l];v=s['languages'][l];profile=json.loads(Path(r['profile']).read_text());instances=profile['checker'].get('instances',1)
            progress=json.loads((Path(r['checkpoint_dir'])/'progress.json').read_text()) if (Path(r['checkpoint_dir'])/'progress.json').exists() else {}
            processed=len(progress.get('processed_batches',[]))
            preserved=r.get('resume_progress',{});ready=v['phase']=='generating' and processed>len(preserved.get('processed_batches',[]))+20
            if not ready:baselines.pop(l,None);observations[l]={'decision':'waiting_for_replay'};continue
            key=r['checkpoint_dir'];base=baselines.get(l)
            if base is None or base['checkpoint']!=key:
                baselines[l]=dict(checkpoint=key,at=now,pairs=v['pairs']);observations[l]={'decision':'measuring_steady_throughput'};continue
            elapsed=now-base['at']
            if elapsed<180:continue
            rate=(v['pairs']-base['pairs'])/elapsed
            ahead=sum(int(p.name.split('.')[0])>=processed for p in Path(key).glob('*.candidates.receipt.json'))
            c=cpus.get(l,{})
            memory=psutil.virtual_memory()
            target,reason=capacity_decision(r['parser_workers'],instances,ahead,c.get('checker',0),c.get('parser',0),host,len(os.sched_getaffinity(0)),max(0,memory.available/2**30-.05*memory.total/2**30),c.pop('_parser_rss_gb',2))
            prior=previous_rates.get(l)
            if target and prior and target['kind']==prior['kind'] and rate<1.1*prior['rate']:
                plateaus[l]=plateaus.get(l,0)+1
                if plateaus[l]>=2:target=None;reason='throughput_gain_plateau'
            else:plateaus[l]=0
            observation=dict(at=s['timestamp'],instances=instances,parser_workers=r['parser_workers'],prepared_batches_ahead=ahead,retained_pairs_per_second=rate,cpu_cores=c,host_cpu_cores=host,decision=reason,proposed_allocation=target)
            observations[l]=observation
            baselines[l]=dict(checkpoint=key,at=now,pairs=v['pairs'])
            if target and action is None and now-last_resize.get(l,0)>300:action=(l,pid,target,observation,rate)
        atomic_json(output/'status.json',dict(status='monitoring',at=s['timestamp'],observations=observations))
        with (output/'history.jsonl').open('a') as f:f.write(json.dumps(dict(at=s['timestamp'],observations=observations))+'\n')
        if action:
            l,pid,target,observation,rate=action
            receipt=resize(args.launch,l,pid,target['instances'],observation,output,parser_workers=target['parser_workers'])
            previous_rates[l]={'kind':target['kind'],'rate':rate};plateaus[l]=0;last_resize[l]=time.time();baselines.pop(l,None)
            print(json.dumps(dict(language=l,allocation=target,supervisor_pid=receipt['supervisor_pid'])),flush=True)
        time.sleep(args.interval)

if __name__=='__main__':main()

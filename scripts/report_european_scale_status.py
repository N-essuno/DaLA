"""Snapshot launched CPU builds and measure observed retained-pair throughput."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def snapshot(launch_path):
    launch=json.loads(launch_path.read_text());now=time.time()
    output=subprocess.check_output(['ps','-eo','pid=,ppid=,args='],text=True)
    processes=[line.strip().split(None,2) for line in output.splitlines() if line.strip()]
    rows={}
    for record in launch['languages']:
        language=record['language'];progress_path=Path(record['checkpoint_dir'])/'progress.json'
        progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
        parent=next((int(pid) for pid,ppid,command in processes if int(ppid)==record.get('supervisor_pid', launch['pid']) and f'--languages {language} ' in command),None)
        workers=sum(int(ppid)==parent and 'multiprocessing.spawn' in command for pid,ppid,command in processes)
        finished=Path(record['output'])/'manifest.json'
        status_path=Path(record.get('language_logs', launch['language_logs']))/(language+'-status.json')
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        phase='complete' if finished.exists() else 'failed' if status.get('exit_code',0) else 'generating' if parent and progress else 'starting' if parent else 'not_running'
        migration_path=progress_path.parent/'migration.json'
        migration=json.loads(migration_path.read_text()) if migration_path.exists() else {}
        previous_progress=record.get('resume_progress', migration.get('previous_progress', {}))
        replaying=bool(parent and previous_progress and len(progress.get('processed_batches',[]))<len(previous_progress.get('processed_batches',[])))
        reconstructed_pairs=progress.get('pairs',0)
        actual_progress=progress
        if replaying:
            phase='replaying' if progress else 'starting'
            progress=previous_progress
        rows[language]=dict(phase=phase,pid=parent,parser_workers=workers,configured_parser_workers=record['parser_workers'],pairs=progress.get('pairs',0),splits=progress.get('splits',{}),processed_batches=len(progress.get('processed_batches',[])),counts=progress.get('counts',{}),exit_code=status.get('exit_code'),count_basis='preserved_checkpoint' if replaying else 'current',reconstructed_pairs=reconstructed_pairs,selected_by_type=actual_progress.get('selected_by_type',{}),timing_seconds=actual_progress.get('timing_seconds',{}),source_batch_count=actual_progress.get('source_batch_count'),total_batches=actual_progress.get('total_batches'))
    result=dict(timestamp=datetime.now(timezone.utc).isoformat(),epoch=now,supervisor_pid=launch['pid'],parser_workers=sum(r['parser_workers'] for r in rows.values()),configured_parser_workers=launch['parser_workers_total'],pairs=sum(r['pairs'] for r in rows.values()),languages=rows)
    history=launch_path.parent/'status-history.jsonl'
    previous=json.loads(history.read_text().splitlines()[-1]) if history.exists() and history.stat().st_size else None
    if previous:
        elapsed=now-previous['epoch'];result['interval_seconds']=elapsed
        for language,r in rows.items():
            delta=r['pairs']-previous['languages'][language]['pairs']
            r['interval_retained_pairs']=delta
            r['pairs_per_second']=round(delta/elapsed,3) if elapsed>0 and delta>=0 else None
    with history.open('a') as f:f.write(json.dumps(result)+'\n')
    (launch_path.parent/'status.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('launch',type=Path);a=p.parse_args()
    result=snapshot(a.launch)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

"""One-shot, user-authorized 128->256 handoff after successful Polish completion.

Run from the repository root. Register this watcher's launch receipt in
current-run.json so a user-requested pause also stops the pending handoff.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path('wiki/artifacts/six-language-expansion')
DEST = ROOT / 'scaled256_v1'
CURRENT = ROOT / 'current-run.json'
OLD = ROOT / 'scaled128_v1'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def status(state, **fields):
    write(DEST / 'handoff-status.json', {**fields, 'at': now(), 'status': state})
    print(now(), state, fields, flush=True)


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return fields[19]  # Linux starttime, field 22, protects against PID reuse.
    except FileNotFoundError:
        return None


def group_members(pgid, parsers_only=False):
    result = []
    for p in Path('/proc').glob('[0-9]*'):
        try:
            pid = int(p.name)
            if os.getpgid(pid) == pgid and identity(pid):
                if not parsers_only or b'spawn_main' in (p / 'cmdline').read_bytes():
                    result.append(pid)
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def capture(path):
    receipt = read(path)
    pid = receipt['pid']
    assert identity(pid) and os.getpgid(pid) == pid
    args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    assert all(item.encode() in args for item in receipt['command'])
    return dict(receipt, starttime=identity(pid))


def stop(receipt):
    pid = receipt['pid']
    assert identity(pid) == receipt['starttime'], 'Process identity changed; refusing signal'
    assert os.getpgid(pid) == pid
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 120
    while group_members(pid):
        if time.monotonic() > deadline:
            raise RuntimeError(f'Process group {pid} did not stop; checkpoints untouched')
        time.sleep(1)


def eligible(record):
    return record.get('exit_code') == 0 and record.get('status') == 'candidate_build_complete_requires_final_audit'


def launch(name, command):
    logfile = f'/tmp/dala-six-scaled256_v1-{name}.log'
    with open(logfile, 'a') as stream:
        process = subprocess.Popen([sys.executable, '-u', *command], stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    path = DEST / f'{name}-launch.json'
    write(path, dict(at=now(), pid=process.pid, command=command, log=logfile))
    current = read(CURRENT)
    current['active_launch_receipts'].append(str(path))
    write(CURRENT, current)
    return process.pid


def main():
    DEST.mkdir(exist_ok=True)
    previous = read('config/languages/is_scaled128_v1.json')
    proposed = read('config/languages/is_scaled256_v1.json')
    previous['build']['parser_processes'] = 256
    previous['build']['checkpoint_dir'] = proposed['build']['checkpoint_dir']
    assert previous == proposed, 'Unexpected profile change'
    assert proposed['build']['target_pairs'] == 478930
    owned = {name: capture(OLD / f'{name}-launch.json') for name in ['is-builds', 'finalizer']}
    write(DEST / 'handoff-inputs.json', owned)
    status('waiting_for_polish_success', target_workers=256)
    while True:
        current = read(CURRENT)
        if current.get('state') != 'running' or current['language_runs']['is'] != 'scaled128_v1':
            status('cancelled_current_run_changed')
            return
        is_status = OLD / 'is-status.json'
        if is_status.exists():
            status('no_restart_icelandic_already_terminal', build=read(is_status))
            return
        pl_status = OLD / 'pl-status.json'
        if pl_status.exists():
            record = read(pl_status)
            if not eligible(record):
                status('blocked_polish_not_successful', build=record)
                return
            # The success receipt follows child exit; also confirm all old Polish workers exited.
            if not group_members(read(OLD / 'pl-builds-launch.json')['pid']):
                break
        time.sleep(15)
    if (OLD / 'is-status.json').exists():
        status('no_restart_icelandic_already_terminal')
        return
    status('stopping_icelandic_and_finalizer')
    stop(owned['finalizer'])
    stop(owned['is-builds'])
    write(DEST / 'preserved-progress.json', read('la_output/is_scaled128_v1_checkpoints/progress.json'))
    status('migrating_verified_checkpoints')
    subprocess.run([sys.executable, '-m', 'scripts.migrate_execution_profile',
                    'config/languages/is_scaled128_v1.json',
                    'config/languages/is_scaled256_v1.json'], check=True)
    current = read(CURRENT)
    current['language_runs']['is'] = 'scaled256_v1'
    current['configured_parser_workers']['is'] = 256
    current['active_parser_workers'].update({'pl': 0, 'is': 0})
    current['status'] = 'starting_icelandic_256_workers'
    current['active_launch_receipts'] = [str(DEST / 'handoff-launch.json')]
    write(CURRENT, current)
    build_pid = launch('is-builds', ['-m', 'scripts.run_six_language_scale', '--languages', 'is',
                                   '--run-id', 'scaled256_v1', '--tools-dir', '/tmp/dala-six-tools',
                                   '--parser-workers', '256'])
    finalizer = [arg.replace('is=scaled128_v1', 'is=scaled256_v1')
                 for arg in owned['finalizer']['command']]
    launch('finalizer', finalizer)
    status('verifying_live_workers', build_pid=build_pid)
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        workers = group_members(build_pid, parsers_only=True)
        if len(workers) == 256:
            evidence = dict(at=now(), live_parser_workers=256, pids=workers, target_pairs=478930)
            write(DEST / 'worker-verification.json', evidence)
            current = read(CURRENT)
            current['active_parser_workers']['is'] = 256
            current['worker_snapshot_at'] = evidence['at']
            current['worker_verification'] = str(DEST / 'worker-verification.json')
            current['status'] = 'running'
            write(CURRENT, current)
            status('complete_256_live_workers', **evidence)
            return
        if (DEST / 'is-status.json').exists():
            status('build_terminal_before_worker_verification', build=read(DEST / 'is-status.json'))
            return
        time.sleep(10)
    raise RuntimeError('Worker verification timed out; inspect running build')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed_requires_inspection', error=repr(exc))
        raise

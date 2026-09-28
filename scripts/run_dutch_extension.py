"""Run the approved Dutch extension, merge, and mechanical assessment in order."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def run():
    status = Path('la_output/dutch_extension_status.json')
    stages = [
        ('generation', [sys.executable, '-m', 'dala.pair_pipeline', '--profile', 'nl_extension_scale', '--max-errors', '1', '--offline', '--output-dir', 'la_output/dutch_extension_scale'], '/tmp/dutch_extension_scale.log'),
        ('merge', [sys.executable, '-m', 'scripts.merge_dutch_extension', 'la_output/dutch_dynaword_uncapped', 'la_output/dutch_extension_scale', 'la_output/dutch_dynaword_extended', '--profile', 'nl_legal_extended'], '/tmp/dutch_extension_merge.log'),
        ('assessment', [sys.executable, '-m', 'scripts.assess_dutch', 'la_output/dutch_dynaword_extended', 'wiki/artifacts/dutch-extended-assessment', '--profile', 'nl_legal_extended', '--sample-size', '200', '--seed', 'dala-dutch-extended-audit-20260922-v1'], '/tmp/dutch_extended_assessment.log'),
    ]
    for root in ('dutch-pilot-assessment', 'dutch-final-assessment', 'dutch-curated-assessment', 'dutch-validation-assessment', 'dutch-full-assessment', 'dutch-extension-pilot-assessment', 'dutch-extension-validation'):
        for name in ('sample.json', 'family-supplement.json'):
            p = Path('wiki/artifacts')/root/name
            if p.exists():
                stages[-1][1].extend(['--exclude-reviewed-sample', str(p)])
    for stage, command, log in stages:
        state = dict(stage=stage, status='running', updated_at=datetime.now(timezone.utc).isoformat(), log=log)
        status.write_text(json.dumps(state, indent=2)+'\n')
        with open(log, 'w') as writer:
            result = subprocess.run(command, stdout=writer, stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(status='failed', returncode=result.returncode, updated_at=datetime.now(timezone.utc).isoformat())
            status.write_text(json.dumps(state, indent=2)+'\n')
            raise SystemExit(result.returncode)
    status.write_text(json.dumps(dict(stage='linguistic_review', status='pending', automated_stages='complete', updated_at=datetime.now(timezone.utc).isoformat(), output='la_output/dutch_dynaword_extended', sample='wiki/artifacts/dutch-extended-assessment/sample.json'), indent=2)+'\n')


if __name__ == '__main__':
    run()

"""Compare serial and checkpointed builders on identical bounded input."""
from pathlib import Path
import json
from dala.profiles import load_profile
from dala.pair_pipeline import _build
from dala.batch_pipeline import build_checkpointed
from dala.language_check import local_server,LanguageCheck
from dala.dataset_review import load_pairs


def main():
    profile=load_profile('en_scale')
    profile['build']['checkpoint_dir']='la_output/scale_smoke_checkpoints'
    profile['build']['document_batch_size']=25
    profile['build']['parser_processes']=2
    with local_server() as url:
        checker=LanguageCheck(url)
        try:
            for function,output in [(_build,'la_output/scale_smoke_serial'),(build_checkpointed,'la_output/scale_smoke_batched'),(build_checkpointed,'la_output/scale_smoke_resumed')]:
                if not Path(output).exists():function(output_dir=output,profile=profile,max_documents=100,offline=True,checker=checker)
        finally:checker.close()
    outputs=[{p['pair_id']:p for p in load_pairs('la_output/'+suffix)} for suffix in ['scale_smoke_serial','scale_smoke_batched','scale_smoke_resumed']]
    assert outputs[0]==outputs[1]==outputs[2]
    r=dict(status='passed',documents=100,pairs=len(outputs[0]),serial_checkpointed_resume_pairs_exactly_equal=True)
    Path('wiki/artifacts/scale-smoke-equivalence.json').write_text(json.dumps(r,indent=2)+'\n');print(r)


if __name__=='__main__':main()

import os
from dotenv import load_dotenv

from llm_gen_eval import run_eval

load_dotenv('../envs.env')
hf_token = os.getenv("HF_TOKEN")

run_eval(
    input_file="../la_output/gen_v2/dala_da_gen_v2_val_formatted_full.csv",
    output_file="predictions/llama-3-8b_2.csv",
    model="meta-llama/Meta-Llama-3-8B",
    hf_token=hf_token,
    hf_model=True,  # Use HuggingFace transformers instead of API
    batch_size=4,   # Process 4 examples at once for faster inference
    max_tokens=100,
    temperature=0.0
)

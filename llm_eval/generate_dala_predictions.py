from llm_gen_eval import run_eval

run_eval(
    input_file="../la_output/gen_v2/dala_da_gen_v2_val_formatted_full.csv",
    output_file="predictions/llama-3-8b.csv",
    model="meta-llama-3-8b",  # Use the API identifier shown
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    max_tokens=100,
    temperature=0.0
)
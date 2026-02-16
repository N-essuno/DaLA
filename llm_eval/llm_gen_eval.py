import os
import pandas as pd
import re
from typing import Tuple, Optional
import openai
import asyncio
from tqdm import tqdm
import torch

def normalize_text(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)  # keep alphanumeric
    return s.strip()


def normalize_entities(s: str) -> set:
    """Split by commas/and and normalize names into a set."""
    s = re.sub(r"\band\b", ",", s, flags=re.IGNORECASE)
    parts = [normalize_text(x) for x in s.split(",")]
    return {p.strip() for p in parts if p.strip()}


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1."""
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Check if normalized prediction exactly matches normalized ground truth."""
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def set_match_score(prediction: str, ground_truth: str) -> float:
    pred_set = normalize_entities(prediction)
    gold_set = normalize_entities(ground_truth)

    em = float(pred_set == gold_set)

    if not pred_set or not gold_set:
        return em, float(pred_set == gold_set)

    common = pred_set & gold_set
    precision = len(common) / len(pred_set)
    recall = len(common) / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if common else 0.0
    return em, f1

def qa_score_single(pred: str, gold: str, args: dict, set_based=False) -> tuple[float, float] | float:
    """Return (EM, F1) for a single QA pair."""
    if not set_based:
        return exact_match_score(pred, gold), f1_score(pred, gold)
    else:
        return set_match_score(pred, gold)


def bleu_score(prediction: str, ground_truth: str) -> float:
    """Compute BLEU score for single prediction and ground truth."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu([gold_tokens], pred_tokens, smoothing_function=smoothie)

def sacrebleu_score(prediction: str, ground_truth: str) -> float:
    """Compute SacreBLEU score for single prediction and ground truth."""
    import sacrebleu
    return sacrebleu.sentence_bleu(prediction, [ground_truth]).score / 100.0

def rouge_l_score(prediction: str, ground_truth: str) -> float:
    """Compute ROUGE-L score for single prediction and ground truth."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'])
    scores = scorer.score(ground_truth, prediction)
    return scores['rougeL'].fmeasure

def meteor_score_hf(prediction: str, ground_truth: str) -> float:
    """Compute METEOR score for single prediction and ground truth. Use HuggingFace's implementation."""
    import evaluate
    meteor = evaluate.load("meteor")
    return meteor.compute(predictions=[prediction], references=[ground_truth])["meteor"]

def meteor_score(prediction: str, ground_truth: str) -> float:
    """Compute METEOR score for single prediction and ground truth."""
    from nltk.translate.meteor_score import meteor_score
    return meteor_score([ground_truth], prediction)

def bert_score(prediction: str, ground_truth: str) -> float:
    """Compute BERTScore for single prediction and ground truth."""
    from bert_score import score
    P, R, F1 = score([prediction], [ground_truth], lang="da")
    return F1.item()

def bert_score_hf(prediction: str, ground_truth: str) -> float:
    """Compute BERTScore for single prediction and ground truth."""
    import evaluate
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=[prediction], references=[ground_truth], lang="da")
    return results["f1"][0]


def error_spot_score(prediction: str, ground_truth: str, args: dict, thr: float = 0.6) -> float:
    # from ..dala.dala_corrupt import SpacyModelSingleton
    # dk_model = SpacyModelSingleton("da_core_news_md")
    #
    # doc_pred = dk_model(prediction)
    # doc_gold = dk_model(ground_truth)

    corruption_type = args.get("corruption_type")
    affected_token_1 = args.get("affected_token_1")
    affected_token_2 = args.get("affected_token_2")
    corrupted_sentence = args.get("corrupted_sentence")

    base_score = sacrebleu_score(prediction, ground_truth)

    if base_score < thr:
        return base_score

    # check if in the gold answer, the affected tokens appear in the same order as ground truth
    if corruption_type == 'flip_neighbours':
        # get tokens
        gold_tokens = normalize_text(ground_truth).split()
        pred_tokens = normalize_text(prediction).split()

        # get indexes of affected tokens in gold, for which indexes are consecutive
        for token in gold_tokens:
            if token == affected_token_1:
                gold_idx1 = gold_tokens.index(token)
                if gold_idx1 < len(gold_tokens) - 1 and gold_tokens[gold_idx1 + 1] == affected_token_2:
                    gold_idx2 = gold_idx1 + 1
                    break
            elif token == affected_token_2:
                gold_idx1 = gold_tokens.index(token)
                if gold_idx1 < len(gold_tokens) - 1 and gold_tokens[gold_idx1 + 1] == affected_token_1:
                    gold_idx2 = gold_idx1 + 1
                    break

        # do the same for pred
        for token in pred_tokens:
            if token == affected_token_1:
                pred_idx1 = pred_tokens.index(token)
                if pred_idx1 < len(pred_tokens) - 1 and pred_tokens[pred_idx1 + 1] == affected_token_2:
                    pred_idx2 = pred_idx1 + 1
                    break
            elif token == affected_token_2:
                pred_idx1 = pred_tokens.index(token)
                if pred_idx1 < len(pred_tokens) - 1 and pred_tokens[pred_idx1 + 1] == affected_token_1:
                    pred_idx2 = pred_idx1 + 1
                    break

        same_order = gold_idx1 < gold_idx2 and pred_idx1 < pred_idx2
        same_index = (gold_idx1 == pred_idx1 and gold_idx2 == pred_idx2)
        return 0.7 * float(same_order) + 0.3 * float(same_index)
    elif corruption_type == 'delete':
        pass
    else:
        # assume affected_token_1 is the correct token in gold
        #   and affected_token_2 is the incorrect token that replaced the original in the corrupted sentence

        gold_tokens = normalize_text(ground_truth).split()
        pred_tokens = normalize_text(prediction).split()

        # get first index of affected_token_1 in gold
        gold_idx = gold_tokens.index(affected_token_1)

        # check if affected_token_1 appears in pred at the same index
        if gold_idx < len(pred_tokens) and pred_tokens[gold_idx] == affected_token_1:
            return 1.0





def evaluate_dataset(path: str, gold_path: str, pred_col: str = "Answer", verbose: str = False,
                     meteor_metric=None, bertscore_hf_metric=None, bertscore_scorer=None) -> dict:
    """Evaluate a CSV or Parquet file with columns Question, Answer, Prediction."""
    if path.endswith(".csv"):
        df = pd.read_csv(path, delimiter=";")
        gold_df = pd.read_csv(gold_path, delimiter=";")
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
        gold_df = pd.read_parquet(gold_path)
    else:
        raise ValueError("File must be .csv or .parquet, provided: " + path)

    # Load metrics if not provided (for backward compatibility)
    if meteor_metric is None or bertscore_hf_metric is None or bertscore_scorer is None:
        import evaluate
        from bert_score import BERTScorer
        if meteor_metric is None:
            meteor_metric = evaluate.load("meteor")
        # if bertscore_hf_metric is None:
        #     bertscore_hf_metric = evaluate.load("bertscore")
        if bertscore_scorer is None:
            bertscore_scorer = BERTScorer(lang="da")

    log_info = []

    ems, f1s, bleus, rouge_ls, meteors, meteors_hf, bertscores, bertscores_hf = [], [], [], [], [], [], [], []
    for (_, row) in df.iterrows():
        gold = gold_df.loc[gold_df["id"] == row["id"], "Answer"].values[0]
        corruption_type = gold_df.loc[gold_df["id"] == row["id"], "corruption_type"].values[0]
        affected_token_1 = gold_df.loc[gold_df["id"] == row["id"], "affected_token_1"].values[0]
        affected_token_2 = gold_df.loc[gold_df["id"] == row["id"], "affected_token_2"].values[0]
        corrupted_sentence = gold_df.loc[gold_df["id"] == row["id"], "Question"].values[0]
        log_info.append((gold, row["Answer"], corrupted_sentence, corruption_type, affected_token_1, affected_token_2))

        if verbose:
            print("Gold:", gold)
            print("Pred:", row["Answer"])
            print(f"Corr: {corrupted_sentence}")
            print(f"Type: {corruption_type}")
            print(f"tok1: {affected_token_1}, tok2: {affected_token_2}\n")


        eval_args = {
            "corruption_type": corruption_type,
            "affected_token_1": affected_token_1,
            "affected_token_2": affected_token_2,
            "corrupted_sentence": corrupted_sentence
        }
        pred = str(row[pred_col])
        em, f1 = qa_score_single(pred, gold, args=eval_args)
        b_score = bleu_score(pred, gold)
        rouge_l = rouge_l_score(pred, gold)
        # meteor = meteor_score(pred, gold)
        # Use pre-loaded metrics for efficiency
        meteor_hf = meteor_metric.compute(predictions=[pred], references=[gold])["meteor"]
        P, R, F1 = bertscore_scorer.score([pred], [gold])
        bertscore = F1.item()
        # bertscore_hf_results = bertscore_hf_metric.compute(predictions=[pred], references=[gold], lang="da")
        # bertscore_hf = bertscore_hf_results["f1"][0]

        ems.append(em)
        f1s.append(f1)
        bleus.append(b_score)
        rouge_ls.append(rouge_l)
        # meteors.append(meteor)
        meteors_hf.append(meteor_hf)
        bertscores.append(bertscore)
        # bertscores_hf.append(bertscore_hf)

    # save log info to a file, note: path file name is the model name. save in current folder with _log.txt suffix
    with open(f"logs/{os.path.splitext(os.path.basename(path))[0]}_log.txt", "w") as f:
        for gold, pred, corr_sent, corr, tok1, tok2 in log_info:
            f.write(f"Gold: {gold}\nPred: {pred}\nCorr: {corr_sent}\nType: {corr}\ntok1: {tok1}, tok2: {tok2}\n\n")

    # save logs as a csv file as well
    log_df = pd.DataFrame(log_info, columns=["target_text_2", "generation_2", "text_2", "corruption_type", "affected_token_1", "affected_token_2"])
    log_df.to_csv(f"logs/{os.path.splitext(os.path.basename(path))[0]}_log.csv", index=False)

    return {
        "EM": sum(ems) / len(ems),
        "F1": sum(f1s) / len(f1s),
        "BLEU": sum(bleus) / len(bleus),
        "ROUGE-L": sum(rouge_ls) / len(rouge_ls),
        # "METEOR": sum(meteors) / len(meteors),
        "METEOR_HF": sum(meteors_hf) / len(meteors_hf),
        "BERTScore": sum(bertscores) / len(bertscores),
        # "BERTScore_HF": sum(bertscores_hf) / len(bertscores_hf),
    }


PROMPT_TEMPLATE = """
Givet følgende sætning på dansk, afgør om den er grammatisk korrekt eller ukorrekt.
- Hvis den er ukorrekt, skal du outputte den rettede version.
- Hvis den er korrekt, skal du outputte den originale sætning.
- Output kun den rettede eller originale sætning — intet andet.

Regler:
- Svar kun på dansk.
- Bevar samme formatering og store/små bogstaver som i den originale sætning.
- Output ingen forklaring.
- Sig ikke, om sætningen er korrekt eller ukorrekt.

\n\nSætning: {question}\nSvar:
"""


async def get_prediction(client, model, identifier, question, expected, max_tokens=100, temperature=0.0):
    prompt = PROMPT_TEMPLATE.format(question=question)
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if response is None or response.choices is None or response.choices[0].message.content is None:
        print(f"Question {identifier}: Response is None")
        prediction = None
    else:
        print(response)
        prediction = response.choices[0].message.content.strip().replace("\n", " ")

    result = {
        "id": identifier,
        "question": question,
        "expected": expected,
        "prediction": prediction,
    }
    if expected is not None:
        result["correct"] = prediction.lower() == expected.lower()
    return result


def get_predictions_hf_batch(model, tokenizer, batch_data, max_tokens=100, temperature=0.0):
    """Get predictions for a batch using HuggingFace transformers (faster)."""
    prompts = [PROMPT_TEMPLATE.format(question=row['question']) for row in batch_data]

    try:
        # Tokenize all prompts at once with padding
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)

        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }

        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["do_sample"] = True
        else:
            generation_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        # Decode all outputs, excluding the prompt tokens
        prompt_lengths = inputs['input_ids'].shape[1]
        predictions = []
        for i, output in enumerate(outputs):
            # Find actual prompt length for this sample (accounting for padding)
            actual_prompt_length = (inputs['input_ids'][i] != tokenizer.pad_token_id).sum().item()
            prediction = tokenizer.decode(
                output[actual_prompt_length:],
                skip_special_tokens=True
            ).strip().replace("\n", " ")
            predictions.append(prediction)

    except Exception as e:
        print(f"Batch error during generation - {str(e)}")
        predictions = [None] * len(batch_data)

    # Build results
    results = []
    for row, prediction in zip(batch_data, predictions):
        result = {
            "id": row['id'],
            "question": row['question'],
            "expected": row['expected'],
            "prediction": prediction,
        }
        if row['expected'] is not None and prediction is not None:
            result["correct"] = prediction.lower() == row['expected'].lower()
        results.append(result)

    return results



def process_hf(input_file: str, output_file: str, model_name: str, max_tokens=100,
                temperature=0.0, batch_size=1, flush=False, hf_token=None) -> str:
    """Process predictions using HuggingFace transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {model_name}...")

    # Prepare loading kwargs with optional token
    load_kwargs = {}
    if hf_token:
        load_kwargs['token'] = hf_token
        print("Using provided HuggingFace token for authentication")

    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device.type in ["cuda", "mps"] else torch.float32,
        low_cpu_mem_usage=True,
        **load_kwargs
    ).to(device)
    model.eval()

    print(f"Model loaded on {device}")

    # Load input data
    if input_file.endswith(".csv"):
        df = pd.read_csv(input_file, delimiter=";")
    elif input_file.endswith(".parquet"):
        df = pd.read_parquet(input_file)
    else:
        raise ValueError("Input file must be .csv or .parquet")

    if output_file is None:
        output_file = model_name.replace("/", "-") + "-predictions.csv"

    print(f"Processing {len(df)} questions...")
    print(f"Writing predictions to {output_file}")
    print(f"Using batch size: {batch_size}")

    with open(output_file, "w") as f:
        f.write("id;Answer\n")

        # Process in batches
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch_df = df.iloc[i:i + batch_size]

            # Prepare batch data
            batch_data = []
            for _, row in batch_df.iterrows():
                expected = row.Answer if hasattr(row, 'Answer') else None
                batch_data.append({
                    'id': row.id,
                    'question': row.Question,
                    'expected': expected
                })

            # Get predictions for entire batch
            results = get_predictions_hf_batch(
                model=model,
                tokenizer=tokenizer,
                batch_data=batch_data,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Write results
            for result in results:
                if result['prediction'] is None:
                    f.write(f"{result['id']};Error\n")
                else:
                    f.write(f"{result['id']};{result['prediction']}\n")

            if flush:
                f.flush()

    return output_file


async def call_api(input_file: str, output_file: str, model: str, base_url: str, api_key: str, max_tokens=100,
                   temperature=0.0, batch_size=20, flush=False) -> str:
    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)

    if input_file.endswith(".csv"):
        df = pd.read_csv(input_file, delimiter=";")
    elif input_file.endswith(".parquet"):
        df = pd.read_parquet(input_file)

    tasks = []
    results = []
    print("Creating tasks...")
    for row in df.itertuples():
        print(f"Processing question: {row.Question}")
        expected = row.Answer if hasattr(row, 'Answer') else None
        tasks.append(
            get_prediction(client=client, model=model, identifier=row.id, question=row.Question, expected=expected,
                           max_tokens=max_tokens, temperature=temperature)
        )
    print(f"Total questions to process: {len(df)}")

    if output_file is None:
        output_file = model.replace("/", "-") + "-predictions.csv"
        print(output_file)
    print(f"Writing predictions to {output_file}")

    if not output_file:
        output_file = model.replace("/", "-") + "-predictions.csv"

    with open(output_file, "w") as f:
        f.write("id;Answer\n")

        for i in tqdm(range(0, len(tasks), batch_size), desc="Processing batches"):
            batch = tasks[i:i + batch_size]
            batch_responses = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_responses)

            for i, response in enumerate(batch_responses):
                if response['prediction'] is None:
                    f.write(f"{response['id']};Error\n")
                else:
                    f.write(f"{response['id']};{response['prediction']}\n")

            if flush:
                f.flush()


def run_eval(input_file: str, output_file: str, model: str, base_url: str = None, api_key: str = None,
             max_tokens=100, temperature=0.0, hf_model: bool = False, batch_size: int = 20, flush: bool = False,
             hf_token: str = None):
    """
    Run evaluation using either HuggingFace models or API.

    Args:
        input_file: Path to input CSV or Parquet file
        output_file: Path to output predictions file
        model: Model name (HF model ID or API model name)
        base_url: API base URL (required if hf_model=False)
        api_key: API key (required if hf_model=False)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        hf_model: If True, use HuggingFace transformers; if False, use API
        batch_size: Batch size for API calls (ignored for HF)
        flush: Whether to flush output after each batch/sample
        hf_token: HuggingFace token for accessing gated/private models (optional)

    Returns:
        Path to output file
    """
    if hf_model:
        print(f"Using HuggingFace model: {model}")
        return process_hf(
            input_file=input_file,
            output_file=output_file,
            model_name=model,
            max_tokens=max_tokens,
            temperature=temperature,
            batch_size=batch_size,
            flush=flush,
            hf_token=hf_token
        )
    else:
        print(f"Using API model: {model}")
        return asyncio.run(
            call_api(
                input_file=input_file,
                output_file=output_file,
                model=model,
                base_url=base_url,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature,
                batch_size=batch_size,
                flush=flush
            )
        )

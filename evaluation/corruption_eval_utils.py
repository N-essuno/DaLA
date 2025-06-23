import os
from typing import List

import requests
import yaml
from tqdm import tqdm
from spacy import Language

import pandas as pd


def corrupt_ud_da(corrupt_func: callable, sentences: List[str], model: Language, flip_prob: float = 1, token_comparison: bool = False) -> pd.DataFrame:
    if token_comparison:
        dala_dict = {"original": [], "corrupted": [], "corruption_type": [], "label": [],
                     "original_token": [], "corrupted_token": []}
    else:
        dala_dict = {"original": [], "corrupted": [], "corruption_type": [], "label": []}

    for sent in sentences:
        dala_dict["original"].append(sent)
        if token_comparison:
            label, result, original_token, corrupted_token = corrupt_func(model, sent, flip_prob, token_comparison=True)
        else:
            label, result = corrupt_func(model, sent, flip_prob)
        if label:
            dala_dict["label"].append("incorrect")
            dala_dict["corrupted"].append(result)
            dala_dict["corruption_type"].append(corrupt_func.__name__)
            if token_comparison:
                # If token comparison is enabled, we can also return the original sentence
                dala_dict["original_token"].append(original_token)
                dala_dict["corrupted_token"].append(corrupted_token)
        else:
            dala_dict["label"].append("correct")
            dala_dict["corrupted"].append(None)
            dala_dict["corruption_type"].append(None)
            if token_comparison:
                # If token comparison is enabled, we can also return the original sentence
                dala_dict["original_token"].append(None)
                dala_dict["corrupted_token"].append(None)
    return pd.DataFrame(dala_dict)

def send_curl_request(url, headers=None, data=None, method="GET"):
    if method.upper() == "GET":
        response = requests.get(url, headers=headers)
    elif method.upper() == "POST":
        response = requests.post(url, headers=headers, data=data)
    elif method.upper() == "PUT":
        response = requests.put(url, headers=headers, data=data)
    elif method.upper() == "DELETE":
        response = requests.delete(url, headers=headers)
    else:
        raise ValueError("Unsupported HTTP method")

    return response


def send_write_assistant_request(laravel_session, laravel_token, XSRF_TOKEN, x_xsrf_token, da_text):
    # Define the URL
    url = 'https://www.writeassistant.com/api/corrections'

    # Define the headers
    headers = {
        'authority': 'www.writeassistant.com',
        'method': 'POST',
        'path': '/api/corrections',
        'scheme': 'https',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7,de;q=0.6,da;q=0.5',
        'Content-Type': 'application/json',
        'Origin': 'https://www.writeassistant.com',
        'Referer': 'https://www.writeassistant.com/',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
        'x-xsrf-token': f'{x_xsrf_token}'
    }

    # Define the cookies (simulate the session and token)
    cookies = {
        'laravel_session': f'{laravel_session}',
        'laravel_token': f'{laravel_token}',
        'XSRF-TOKEN': f'{XSRF_TOKEN}'
    }

    # Define the payload (JSON data)
    data = {
        'context': {
            'la_test': {
                'text': f'{da_text}',
                'type': 'P'
            }
        },
        'settings': {
            'languageFrom': 'da',
            'languageTo': 'da',
            'language_settings': {}
        }
    }

    # save_curl_request(url, headers, cookies, data, filename="send_request.sh")

    # Send the POST request
    response = requests.post(url, headers=headers, cookies=cookies, json=data)

    return response


def get_error_types(response) -> List[str]:
    try:
        error_type_list = []
        corrections = response.json()['result']['la_test']['corrections']
        for correction in corrections:
            if 'properties' in correction and 'subtitle' in correction['properties']:
                error_type = correction['properties']['subtitle']
                error_type_list.append(error_type)

        return error_type_list
    except (KeyError, IndexError):
        return []

def get_write_assistant_tokens() -> (str, str):
    with open('tokens.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Assign to variables
    laravel_session = config.get('laravel_session')
    laravel_token = config.get('laravel_token')
    XSRF_TOKEN = config.get('XSRF_TOKEN')
    x_xsrf_token = config.get('x_xsrf_token')

    return laravel_session, laravel_token, XSRF_TOKEN, x_xsrf_token


def evaluate_corruptions(corrupted_df: pd.DataFrame, errors_to_detect: List[str],
                         not_detected_filename: str = "not_detected", token_comparison: bool = False):
    index_reached = 0

    tp = 0
    fp = 0

    laravel_session, laravel_token, XSRF_TOKEN, x_xsrf_token = get_write_assistant_tokens()

    not_detected_df = pd.DataFrame(columns=["sentence", "error_types"])
    detected_df = pd.DataFrame(columns=["sentence", "error_types"])

    #  if a DataFrame is provided, get corrupted_token, original_token and corrupted sentence as lists from it
    incorrect_df = corrupted_df[corrupted_df["label"] == "incorrect"]
    corrupted_sentences = incorrect_df["corrupted"].tolist()
    total_sentences = len(corrupted_sentences)
    if token_comparison:
        corrupted_tokens = incorrect_df["corrupted_token"].tolist()
        original_tokens = incorrect_df["original_token"].tolist()
        iteration_set = zip(corrupted_sentences, original_tokens, corrupted_tokens)
    else:
        iteration_set = zip(corrupted_sentences, [None] * len(corrupted_sentences), [None] * len(corrupted_sentences))

    for sentence, original_token, corrupted_token in tqdm(iteration_set, desc="Evaluating corruptions", total=total_sentences):
        # Put space after " character to avoid wrong detection
        sentence = (sentence
                    .replace('"', ' " ')
                    .replace("(", "( ")
                    .replace(")", " )")
                    )

        response = send_write_assistant_request(laravel_session, laravel_token, XSRF_TOKEN, x_xsrf_token, sentence)
        error_types = get_error_types(response)
        if response.status_code != 200:
            tqdm.write(f"Warning - Response status: {response.status_code}, interrupting the evaluation.")
            with open("status_error.txt", "a") as f:
                f.write(f"============= Sentence: {sentence}\n\n")
                f.write(f"============= Full response: \n\n{response.text}\n\n")
            break

        # Check if at least one error to detect is present in the error types
        if any(error in error_types for error in errors_to_detect):
            tp += 1
            # Add detected sentence to the DataFrame
            error_df = pd.DataFrame(
                [[sentence, error_types, original_token, corrupted_token]],
                columns=["sentence", "error_types", "original_token", "corrupted_token"]
            )
            detected_df = pd.concat([detected_df, error_df], ignore_index=True)
        else:
            fp += 1
            # Add not detected sentence to the DataFrame
            error_df = pd.DataFrame(
                [[sentence, error_types, original_token, corrupted_token]],
                columns=["sentence", "error_types", "original_token", "corrupted_token"]
            )
            not_detected_df = pd.concat([not_detected_df, error_df], ignore_index=True)

        index_reached += 1
        # Sleep to avoid overwhelming the server
        # sleep(2)

    # Create the not_detected directory if it doesn't exist
    not_detected_dir = "./not_detected"
    os.makedirs(not_detected_dir, exist_ok=True)

    # Create the detected directory if it doesn't exist
    detected_dir = "./detected"
    os.makedirs(detected_dir, exist_ok=True)

    # Append not detected sentences to a xlsx file
    not_detected_df.to_excel(f"{not_detected_dir}/wa_{not_detected_filename}_not_detected.xlsx", index=False)

    # Append detected sentences to a xlsx file
    detected_df.to_excel(f"{detected_dir}/wa_{not_detected_filename}_detected.xlsx", index=False)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    return total_sentences, tp, fp, precision, index_reached

def get_corrupted_list(corruption_fn: callable, sentences: List[str], model: Language):
    corruption_df = (corrupt_ud_da(corruption_fn, sentences, model)
                     .sort_values(by=["label"], ascending=False, ignore_index=True))
    corrupted_sentences = corruption_df[corruption_df["label"] == "incorrect"]["corrupted"].tolist()
    return corrupted_sentences


def save_curl_request(url, headers, cookies, data, filename="request.sh"):
    curl_command = f'curl \'{url}\' \\\n'

    for header, value in headers.items():
        curl_command += f'  -H \'{header}: {value}\' \\\n'

    for cookie, value in cookies.items():
        curl_command += f'  -b \'{cookie}={value}\' \\\n'

    curl_command += f'  --data-raw \"{data}\"\n'

    # Save the curl command to a shell script file
    with open(filename, "w") as file:
        file.write(curl_command)

    print(f"Saved the curl request to {filename}")
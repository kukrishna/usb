import os
from copy import deepcopy

import jsonlines
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import argparse
import re

MODEL_NAME = "google/flan-t5-base"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def process_multisentence_compression(dp):
    dp = deepcopy(dp)
    task_name="multisentence_compression"
    PROMPT_STR = "Summarize the following content in a single line : "

    dp["task"] = task_name
    input_string = " ".join(dp["input_lines"])
    output_string = " ".join(dp["output_lines"])
    input_string = f"{PROMPT_STR} {input_string}"
    dp["input_string"] = input_string
    dp["output_string"] = output_string

    return [dp]


def process_abstractive_summarization(dp):
    dp = deepcopy(dp)
    task_name="abstractive_summarization"
    PROMPT_STR = "Summarize the following content : "

    dp["task"] = task_name
    input_string = " ".join(dp["input_lines"])
    output_string = " ".join(dp["output_lines"])
    input_string = f"{PROMPT_STR} {input_string}"
    dp["input_string"] = input_string
    dp["output_string"] = output_string

    return [dp]


def process_fixing_factuality(dp):
    dp = deepcopy(dp)
    task_name="fixing_factuality"
    PROMPT_STR = "Rewrite the given summary of the content to make it factually correct. "

    dp["task"] = task_name
    content_string = " ".join(dp["input_lines"])
    before_summary = dp["initial_summary"]
    # putting the summmary first so that it is never truncated.
    # although there shouldn't be any truncation for this task in general.
    input_string = f"{PROMPT_STR} Summary: {before_summary} Content: {content_string}"
    output_string = dp["fixed_summary"]
    dp["input_string"] = input_string
    dp["output_string"] = output_string

    return [dp]


def process_unsupported_span_prediction(dp):
    dp = deepcopy(dp)
    task_name="unsupported_span_prediction"
    PROMPT_STR = "Annotate parts of the summary which are not supported by evidence from the content. "

    dp["task"] = task_name
    content_string = " ".join(dp["input_lines"])
    before_summary = dp["summary"]
    # putting the summmary first so that it is never truncated.
    # although there shouldn't be any truncation for this task in general.
    input_string = f"{PROMPT_STR} Summary: {before_summary} Content: {content_string}"
    output_string = dp["annotated_summary"]
    dp["input_string"] = input_string
    dp["output_string"] = output_string

    return [dp]

def process_topicbased_summarization(dp):
    dp = deepcopy(dp)
    task_name="topicbased_summarization"
    PROMPT_STR = "Summarize the given content for the following topic : "

    dp["task"] = task_name
    input_string = " ".join(dp["input_lines"])
    output_string = " ".join(dp["output_lines"])
    topic_name = dp["topic_name"]
    input_string = f"{PROMPT_STR} TOPIC: {topic_name} CONTENT: {input_string}"
    dp["input_string"] = input_string
    dp["output_string"] = output_string

    return [dp]


def process_factuality_classification(dp):
    dp = deepcopy(dp)
    task_name="factuality_classification"
    PROMPT_STR = "Is there sufficient evidence for the summary in the content? : "

    dp["task"] = task_name
    input_lines = dp["input_lines"]
    if "summary_sent" not in dp:
        pdb.set_trace()

    summary = dp["summary_sent"]
    content = content = " ".join(input_lines)

    label = dp["label"]
    label_str = "Yes" if label==0 else "No"
    input_string = f"{PROMPT_STR} SUMMARY: {summary} CONTENT: {content}"
    dp["input_string"] = input_string
    dp["output_string"] = label_str

    return [dp]


def process_extractive_summarization(dp, input_maxtoklen):
    dp = deepcopy(dp)
    task_name="extractive_summarization"
    PROMPT_STR = "For each sentence, predict if it is important. "

    this_out_dps=[]

    dp["task"] = task_name
    input_lines = deepcopy(dp["input_lines"])
    labels = deepcopy(dp["labels"])

    sub_index=0

    while len(input_lines)>0:
        towrite_input_lines=[]
        towrite_labels=[]
        towrite_input_string=f"{PROMPT_STR}"
        towrite_output_string=""
        towrite_numtoks=len(tok.encode(towrite_input_string, add_special_tokens=False))

        while len(input_lines)>0:
            cur_line = input_lines[0]
            cur_label = labels[0]
            sent_index = len(towrite_input_lines)
            token_ids = tok.encode(f"SENT{sent_index} {cur_line}", add_special_tokens=False)

            if towrite_numtoks + len(token_ids) < input_maxtoklen:
                towrite_input_string = f"{towrite_input_string} SENT{sent_index} {cur_line}"
                if cur_label:
                    towrite_output_string = f"{towrite_output_string} SENT{sent_index} Yes"
                else:
                    towrite_output_string = f"{towrite_output_string} SENT{sent_index} No"
                towrite_input_lines.append(cur_line)
                towrite_labels.append(cur_label)
                towrite_numtoks += len(token_ids)

                input_lines = input_lines[1:]
                labels = labels[1:]

            else:
                break

        if len(towrite_input_lines)>0:
            dp_to_add = {
                "task": task_name,
                "input_lines": towrite_input_lines,
                "labels": towrite_labels,
                "input_string": towrite_input_string.strip(),
                "output_string": towrite_output_string.strip(),
                "sub_index": sub_index
            }

            if "id" in dp:
                dp_to_add["id"] = dp["id"]

            this_out_dps.append(dp_to_add)
            sub_index+=1

    return this_out_dps



def process_evidence_extraction(dp, input_maxtoklen):
    dp = deepcopy(dp)
    task_name="evidence_extraction"
    PROMPT_STR = "For each sentence in the content, predict if it provides any evidence for the claim. "

    this_out_dps=[]

    summary_lines = deepcopy(dp["summary_lines"])
    evidence_labels = deepcopy(dp["evidence_labels"])

    for summ_idx, (ev_labs, summary_line) in enumerate(zip(evidence_labels, summary_lines)):
        input_lines = deepcopy(dp["input_lines"])

        sub_index=0
        first_line_index=0

        while len(input_lines)>0:
            towrite_input_lines=[]
            towrite_labels=[]
            towrite_input_string=f"{PROMPT_STR} CLAIM: {summary_line} CONTENT: "
            towrite_output_string=""
            towrite_numtoks=len(tok.encode(towrite_input_string, add_special_tokens=False))
            initial_prefix_toklen=towrite_numtoks

            while len(input_lines)>0:
                sent_index = len(towrite_input_lines)
                cur_line = input_lines[0]
                token_ids = tok.encode(f"SENT{sent_index} {cur_line}", add_special_tokens=False,
                                       truncation=True, max_length=input_maxtoklen-initial_prefix_toklen-1)
                # in ridiculous cases where a single sentence is of length MAXTOK-initial_prefix_length, truncate it anyway

                if towrite_numtoks + len(token_ids) < input_maxtoklen:
                    towrite_input_string = f"{towrite_input_string} SENT{sent_index} {cur_line}"
                    if first_line_index in ev_labs:
                        towrite_output_string = f"{towrite_output_string} SENT{sent_index} Yes"
                        towrite_labels.append(sent_index)
                        # remember it's not first_line_index
                    else:
                        towrite_output_string = f"{towrite_output_string} SENT{sent_index} No"
                    towrite_input_lines.append(cur_line)
                    towrite_numtoks += len(token_ids)

                    input_lines = input_lines[1:]
                    first_line_index +=1
                else:
                    break

            if len(towrite_input_lines)>0:
                dp_to_add = {
                    "task": task_name,
                    "input_lines": towrite_input_lines,
                    "labels": towrite_labels,
                    "input_string": towrite_input_string.strip(),
                    "output_string": towrite_output_string.strip(),
                    "sub_index": sub_index,
                    "summ_idx": summ_idx
                }
                if "id" in dp:
                    dp_to_add["id"] = dp["id"]

                this_out_dps.append(dp_to_add)

                sub_index+=1

    return this_out_dps


if __name__=="__main__":

    np.random.seed(1337)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-folder-root", type=str, required=True)
    parser.add_argument("--output-folder-root", type=str, required=True)
    parser.add_argument("--input-maxtoklen", type=int, default=1024)
    args = parser.parse_args()

    dataset_folder_root = args.dataset_folder_root
    output_folder_root = args.output_folder_root
    input_maxtoklen = args.input_maxtoklen

    os.makedirs(output_folder_root, exist_ok=True)

    task_names = os.listdir(dataset_folder_root)

    ALL_SPLITS = ["train", "validation", "test"]

    for task_name in task_names:
        if task_name=="canonical":
            pass

        elif task_name=="multisentence_compression":
            os.mkdir(f"{output_folder_root}/{task_name}")
            for split_name in ALL_SPLITS:
                fpath = f"{dataset_folder_root}/{task_name}/{split_name}.jsonl"
                dps = list(jsonlines.open(fpath))

                out_dps = []

                for dp in dps:
                    new_dps = process_multisentence_compression(dp)
                    out_dps.extend(new_dps)

                with jsonlines.open(f"{output_folder_root}/{task_name}/{split_name}.jsonl", "w") as w:
                    for dp in out_dps:
                        w.write(dp)

        elif task_name=="abstractive_summarization":
            os.mkdir(f"{output_folder_root}/{task_name}")
            for split_name in ALL_SPLITS:
                fpath = f"{dataset_folder_root}/{task_name}/{split_name}.jsonl"
                dps = list(jsonlines.open(fpath))

                out_dps = []

                for dp in dps:
                    new_dps = process_abstractive_summarization(dp)
                    out_dps.extend(new_dps)

                with jsonlines.open(f"{output_folder_root}/{task_name}/{split_name}.jsonl", "w") as w:
                    for dp in out_dps:
                        w.write(dp)

        elif task_name=="fixing_factuality":
            os.mkdir(f"{output_folder_root}/{task_name}")
            PROMPT_STR = "Rewrite the given summary of the content to make it factually correct. "
            for split_name in ALL_SPLITS:
                fpath = f"{dataset_folder_root}/{task_name}/{split_name}.jsonl"
                dps = list(jsonlines.open(fpath))

                out_dps = []

                for dp in dps:
                    new_dps = process_fixing_factuality(dp)
                    out_dps.extend(new_dps)

                with jsonlines.open(f"{output_folder_root}/{task_name}/{split_name}.jsonl", "w") as w:
                    for dp in out_dps:
                        w.write(dp)

        elif task_name=="unsupported_span_prediction":
            os.mkdir(f"{output_folder_root}/{task_name}")
            PROMPT_STR = "Annotate parts of the summary which are not supported by evidence from the content. "
            for split_name in ALL_SPLITS:
                fpath = f"{dataset_folder_root}/{task_name}/{split_name}.jsonl"
                dps = list(jsonlines.open(fpath))

                out_dps = []

                for dp in dps:
                    new_dps = process_unsupported_span_prediction(dp)
                    out_dps.extend(new_dps)

                with jsonlines.open(f"{output_folder_root}/{task_name}/{split_name}.jsonl", "w") as w:
                    for dp in out_dps:
                        w.write(dp)


        elif task_name=="topicbased_summarization":
            os.mkdir(f"{output_folder_root}/{task_name}")
            for split_name in ALL_SPLITS:
                fpath = f"{dataset_folder_root}/{task_name}/{split_name}.jsonl"
                dps = list(jsonlines.open(fpath))

                out_dps = []

                for dp in dps:
                    new_dps = process_topicbased_summarization(dp)
                    out_dps.extend(new_dps)

                with jsonlines.open(f"{output_folder_root}/{task_name}/{split_name}.jsonl", "w") as w:
                    for dp in out_dps:
                        w.write(dp)


        elif task_name=="factuality_classification":
            os.mkdir(f"{output_folder_root}/{task_name}")
            for split_name in ALL_SPLITS:
                fpath = f"{dataset_folder_root}/{task_name}/{split_name}.jsonl"
                dps = list(jsonlines.open(fpath))

                out_dps = []

                for dp in dps:
                    new_dps = process_factuality_classification(dp)
                    out_dps.extend(new_dps)

                with jsonlines.open(f"{output_folder_root}/{task_name}/{split_name}.jsonl", "w") as w:
                    for dp in out_dps:
                        w.write(dp)


        elif task_name=="extractive_summarization":
            os.mkdir(f"{output_folder_root}/{task_name}")
            for split_name in ALL_SPLITS:
                fpath = f"{dataset_folder_root}/{task_name}/{split_name}.jsonl"
                dps = list(jsonlines.open(fpath))

                out_dps = []

                for dp in dps:
                    new_dps = process_extractive_summarization(dp, input_maxtoklen)
                    out_dps.extend(new_dps)


                with jsonlines.open(f"{output_folder_root}/{task_name}/{split_name}.jsonl", "w") as w:
                    for dp in out_dps:
                        w.write(dp)

        elif task_name=="evidence_extraction":
            os.mkdir(f"{output_folder_root}/{task_name}")
            for split_name in ALL_SPLITS:
                fpath = f"{dataset_folder_root}/{task_name}/{split_name}.jsonl"
                dps = list(jsonlines.open(fpath))

                out_dps = []

                for dp in tqdm(dps):
                    new_dps = process_evidence_extraction(dp, input_maxtoklen)
                    out_dps.extend(new_dps)

                with jsonlines.open(f"{output_folder_root}/{task_name}/{split_name}.jsonl", "w") as w:
                    for dp in out_dps:
                        w.write(dp)



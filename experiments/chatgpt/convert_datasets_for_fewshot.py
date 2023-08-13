import os
import pdb
from collections import defaultdict
from copy import deepcopy

import jsonlines
import tiktoken
from tqdm import tqdm
import numpy as np
import argparse
import json


ALL_PROMPT_STRS = {
"multisentence_compression": "Summarize the following content in a single line.",
"abstractive_summarization": "Draft a summary for the given document.",
"fixing_factuality": "The summary might be incorrect. How would you rewrite it to make it factually accurate?",
"unsupported_span_prediction": "Go over the given summary carefully, and regenerate it while surrounding any parts which are not supported by the content using [] and [/] tags.",
"topicbased_summarization": "Create a short summary of the given content that touches upon information which fall under the specified topic.",
"factuality_classification": "Decide if the following summary is consistent with the corresponding content. Note that consistency means all information in the summary is supported by the content. Answer yes or no.",
"extractive_summarization": "For the task of extractive summarization, list all the SENTs of the content which would be included in its summary.",
"evidence_extraction": "Below is a claim along with its corresponding content. Identify and list all the sentences within the content that partially or entirely support the claim.",
}


# computed based on the training set
MAX_OUTPUT_LENS = {
"abstractive_summarization": 557,
"fixing_factuality": 211,
"topicbased_summarization": 368,
"multisentence_compression": 211,
"unsupported_span_prediction": 215,
"factuality_classification": 1
}



def truncate(input_lines, tok, maxlen):
    allowed_input_lines = []
    removed_input_lines = []
    len_sofar = 0
    for line in input_lines:
        tokids = tok.encode(line)
        len_sofar+=len(tokids)
        if len_sofar<=maxlen:
            allowed_input_lines.append(line)
        else:
            removed_input_lines.append(line)

    return allowed_input_lines, removed_input_lines

def getlength(string, tok):
    tokids = tok.encode(string)
    return len(tokids)


def process_cluster2sent(dp, tok, maxlen):
    input_lines = dp["input_lines"]
    output_string = " ".join(dp["output_lines"])

    nontruncable_str = f"CONTENT: \nSUMMARY:"
    nontruncable_len = getlength(nontruncable_str, tok)
    input_lines, removed_lines = truncate(input_lines, tok, maxlen=maxlen-nontruncable_len)
    input_string = " ".join(input_lines)

    input_string = f"CONTENT: {input_string}\nSUMMARY:"
    return [{"input_string":input_string, "output_string":output_string}]

def process_full2full(dp, tok, maxlen):
    input_lines = dp["input_lines"]
    output_string = " ".join(dp["output_lines"])

    nontruncable_str = f"CONTENT: \nSUMMARY:"
    nontruncable_len = getlength(nontruncable_str, tok)
    input_lines, removed_lines = truncate(input_lines, tok, maxlen=maxlen-nontruncable_len)
    input_string = " ".join(input_lines)

    input_string = f"CONTENT: {input_string}\nSUMMARY:"
    return [{"input_string":input_string, "output_string":output_string}]

def process_fixfactuality(dp, tok, maxlen):
    input_lines = dp["input_lines"]
    before_summary = dp["initial_summary"]
    # putting the summmary first so that it is never truncated.
    # although there shouldn't be any truncation for this task in general.

    nontruncable_str = f"CONTENT: \nSUMMARY: {before_summary}\nFIXED SUMMARY:"
    nontruncable_len = getlength(nontruncable_str, tok)
    input_lines, removed_lines = truncate(input_lines, tok, maxlen=maxlen-nontruncable_len)
    content_string = " ".join(input_lines)

    input_string = f"CONTENT: {content_string}\nSUMMARY: {before_summary}\nFIXED SUMMARY:"
    output_string = dp["fixed_summary"]
    return [{"input_string":input_string, "output_string":output_string}]


def process_unsup_spans(dp, tok, maxlen):
    input_lines = dp["input_lines"]
    before_summary = dp["summary"]
    # putting the summmary first so that it is never truncated.
    # although there shouldn't be any truncation for this task in general.

    nontruncable_str = f"CONTENT: \nSUMMARY: {before_summary}\nOUTPUT:"
    nontruncable_len = getlength(nontruncable_str, tok)
    input_lines, removed_lines = truncate(input_lines, tok, maxlen=maxlen-nontruncable_len)
    content_string = " ".join(input_lines)

    input_string = f"CONTENT: {content_string}\nSUMMARY: {before_summary}\nOUTPUT:"
    output_string = dp["annotated_summary"]
    return [{"input_string":input_string, "output_string":output_string}]

def process_full2section(dp, tok, maxlen):
    input_lines = dp["input_lines"]
    output_string = " ".join(dp["output_lines"])
    topic_name = dp["topic_name"]

    nontruncable_str = f"TOPIC: {topic_name}\nCONTENT: \nSUMMARY:"
    nontruncable_len = getlength(nontruncable_str, tok)
    input_lines, removed_lines = truncate(input_lines, tok, maxlen=maxlen-nontruncable_len)
    input_string = " ".join(input_lines)

    input_string = f"TOPIC: {topic_name}\nCONTENT: {input_string}\nSUMMARY:"
    return [{"input_string":input_string, "output_string":output_string}]

def process_factuality_classification(dp, tok, maxlen):
    input_lines = dp["input_lines"]
    if "summary_sent" not in dp:
        pdb.set_trace()
    summary = dp["summary_sent"]
    label = dp["label"]
    label_str = " Yes" if label==0 else " No" # the space is needed because the predict_classes script expects it
    output_string = label_str

    nontruncable_str = f"CONTENT: \nSUMMARY: {summary}\nLABEL:"
    nontruncable_len = getlength(nontruncable_str, tok)
    input_lines, removed_lines = truncate(input_lines, tok, maxlen=maxlen-nontruncable_len)
    content = " ".join(input_lines)

    input_string = f"CONTENT: {content}\nSUMMARY: {summary}\nLABEL:"
    return [{"input_string":input_string, "output_string":output_string}]


def process_important_classification(dp, tok, maxlen):
    this_out_dps=[]

    input_lines = deepcopy(dp["input_lines"])
    labels = deepcopy(dp["labels"])

    sub_index=0

    while len(input_lines)>0:
        towrite_input_lines=[]
        towrite_labels=[]
        towrite_input_string=""
        towrite_output_string=""
        towrite_numtoks=len(tok.encode(towrite_input_string))

        while len(input_lines)>0:
            cur_line = input_lines[0]
            cur_label = labels[0]
            sent_index = len(towrite_input_lines)
            token_ids = tok.encode(f"SENT{sent_index} {cur_line}")
            if cur_label:
                output_ids = tok.encode(f"SENT{sent_index}")
            else:
                output_ids = []

            # the plus ones are for spaces potentially
            if towrite_numtoks + len(token_ids) + 1 + len(output_ids) + 1 < maxlen:
                towrite_input_string = f"{towrite_input_string} SENT{sent_index} {cur_line}"
                if cur_label:
                    towrite_output_string = f"{towrite_output_string} SENT{sent_index}"
                else:
                    pass
                towrite_input_lines.append(cur_line)
                towrite_labels.append(cur_label)
                towrite_numtoks += len(token_ids) + 1 + len(output_ids) + 1

                input_lines = input_lines[1:]
                labels = labels[1:]

            else:
                break

        if len(towrite_input_lines)>0:
            towrite_input_string = towrite_input_string.strip()
            towrite_input_string = f"CONTENT: {towrite_input_string}\nLABELS:"
            towrite_output_string = towrite_output_string.strip()

            this_out_dps.append({
                "input_string": towrite_input_string,
                "output_string": towrite_output_string,
                "input_lines": towrite_input_lines,
                "labels": towrite_labels,
                "sub_index": sub_index
            })
            sub_index+=1

    return this_out_dps



def process_evidence_extraction(dp, tok, maxlen):

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
            towrite_input_string=f"CLAIM: {summary_line}\nCONTENT: "
            towrite_output_string=""
            towrite_numtoks=len(tok.encode(towrite_input_string))
            initial_prefix_toklen=towrite_numtoks

            while len(input_lines)>0:
                sent_index = len(towrite_input_lines)
                cur_line = input_lines[0]
                token_ids = tok.encode(f"SENT{sent_index} {cur_line}")
                # in ridiculous cases where a single sentence is of length MAXTOK-initial_prefix_length, truncate it anyway
                token_ids = token_ids[:maxlen-initial_prefix_toklen-1]

                if first_line_index in ev_labs:
                    output_tokids = tok.encode(f"SENT{sent_index}")
                else:
                    output_tokids = []

                if towrite_numtoks + len(token_ids) + 1 + len(output_tokids) + 1 < maxlen:
                    towrite_input_string = f"{towrite_input_string} SENT{sent_index} {cur_line}"
                    if first_line_index in ev_labs:
                        towrite_output_string = f"{towrite_output_string} SENT{sent_index}"
                        towrite_labels.append(sent_index)
                        # remember it's not first_line_index
                    else:
                        pass # do not write if it's not evidence
                    towrite_input_lines.append(cur_line)
                    towrite_numtoks += len(token_ids) + 1 + len(output_tokids) + 1

                    input_lines = input_lines[1:]
                    first_line_index +=1
                else:
                    break

            if len(towrite_input_lines)>0:
                towrite_input_string = towrite_input_string.strip()
                towrite_input_string = f"{towrite_input_string}\nLABELS:"
                towrite_output_string = towrite_output_string.strip()

                this_out_dps.append({
                    "input_string": towrite_input_string,
                    "output_string": towrite_output_string,
                    "input_lines": towrite_input_lines,
                    "labels": towrite_labels,
                    "sub_index": sub_index,
                    "summ_idx": summ_idx
                })
                sub_index+=1

    return this_out_dps





ALL_DSCREATE_FUNCS = {
"multisentence_compression": process_cluster2sent,
"abstractive_summarization": process_full2full,
"fixing_factuality": process_fixfactuality,
"unsupported_span_prediction": process_unsup_spans,
"topicbased_summarization": process_full2section,
"factuality_classification": process_factuality_classification,
"extractive_summarization": process_important_classification,
"evidence_extraction": process_evidence_extraction,
}


if __name__=="__main__":

    np.random.seed(1337)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-folder-root", type=str, required=True)
    parser.add_argument("--output-folder-root", type=str, required=True)
    parser.add_argument("--input-maxtoklen", type=int, default=4096)
    parser.add_argument("--model-name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--num-shots", type=int, default=4)
    parser.add_argument("--shot-maxtoklen", type=int, default=512)
    parser.add_argument("--tok-eps", type=int, default=8)
    parser.add_argument('--dummyrun', action='store_true')

    args = parser.parse_args()

    dataset_folder_root = args.dataset_folder_root
    output_folder_root = args.output_folder_root
    input_maxtoklen = args.input_maxtoklen
    model_name = args.model_name
    num_shots = args.num_shots
    shot_maxtoklen = args.shot_maxtoklen
    TOK_EPS=args.tok_eps
    is_dummyrun = args.dummyrun

    tok = tiktoken.encoding_for_model(model_name)

    os.makedirs(output_folder_root, exist_ok=True)

    task_names = os.listdir(dataset_folder_root)
    task_names = sorted(task_names)

    short_training_dps = defaultdict(list)
    short_training_dps_lengths = defaultdict(list)

    for task_name in task_names:
        dscreate_func = ALL_DSCREATE_FUNCS[task_name]
        prompt_val = ALL_PROMPT_STRS[task_name]

        fpath = f"{dataset_folder_root}/{task_name}/train.jsonl"
        train_dps = list(jsonlines.open(fpath))

        for dp in tqdm(train_dps):
            dp = deepcopy(dp)
            dp["task"] = task_name

            new_dps = dscreate_func(dp, tok, maxlen=999999) # no truncation needed here. we are only doing this to select the short ones
            new_dps = new_dps[:1]

            for onedp in new_dps:
                input_string = onedp["input_string"]
                output_string = onedp["output_string"]
                inp_numtoks = tok.encode(input_string)
                out_numtoks = tok.encode(output_string)
                total_numtoks = len(inp_numtoks)+len(out_numtoks)

                if total_numtoks<=shot_maxtoklen:
                    short_training_dps[task_name].append(deepcopy(onedp))
                    short_training_dps_lengths[task_name].append(total_numtoks)


    print("Found the following number of short examples for each task : ")
    for (k,v) in short_training_dps.items():
        print(k,len(v))

    total_input_toklen = defaultdict(lambda: defaultdict(int))

    for task_name in task_names:

        print("CREATING DATASET FOR TASK = ", task_name)

        dscreate_func = ALL_DSCREATE_FUNCS[task_name]
        prompt_val = ALL_PROMPT_STRS[task_name]

        os.makedirs(f"{output_folder_root}/{task_name}", exist_ok=True)

        for split_name in ["validation", "test"]:
            fpath = f"{dataset_folder_root}/{task_name}/{split_name}.jsonl"
            if not os.path.exists(fpath):
                continue

            dps = list(jsonlines.open(fpath))

            # if this is a dummy run just take 30 documents
            if is_dummyrun:
                dps = dps[:30]

            out_dps = []

            for dp in tqdm(dps):

                # this part prepares the instructions and fewshot examples
                init_input_string = prompt_val+"\n"

                num_to_choose_from = len(short_training_dps[task_name])
                choices = np.random.permutation(num_to_choose_from)[:num_shots]

                for chosen_idx in choices:
                    ex_dp = short_training_dps[task_name][chosen_idx]
                    ex_toklen = short_training_dps_lengths[task_name][chosen_idx]
                    ex_inp = ex_dp["input_string"]
                    ex_out = ex_dp["output_string"]
                    init_input_string = f"{init_input_string}{ex_inp}{ex_out}\n"

                len_sofar = getlength(init_input_string, tok)

                dp = deepcopy(dp)

                if task_name in MAX_OUTPUT_LENS.keys():
                    new_dps = dscreate_func(dp, tok, maxlen=input_maxtoklen-len_sofar-TOK_EPS-MAX_OUTPUT_LENS[task_name])
                else:
                    new_dps = dscreate_func(dp, tok, maxlen=input_maxtoklen-len_sofar-TOK_EPS)

                for onedp in new_dps:
                    input_string = onedp["input_string"]
                    output_string = onedp["output_string"]

                    final_input_string = f"{init_input_string}{input_string}"
                    final_output_string = output_string

                    onedp["input_string"] = final_input_string
                    onedp["output_string"] = final_output_string
                    onedp["task"] = task_name

                    final_total_len = getlength(onedp["input_string"]+onedp["output_string"], tok)
                    if final_total_len>input_maxtoklen:
                        print("found a very long string with length=",final_total_len)
                        zero_shot_inpstr = f"{prompt_val}\n{input_string}"
                        retried_input_len = getlength(f"{zero_shot_inpstr}{onedp['output_string']}", tok)
                        if retried_input_len<=input_maxtoklen:
                            onedp["input_string"] = zero_shot_inpstr
                            print("falling back to zero shot with length=", retried_input_len)
                            total_input_toklen[task_name][split_name]+=retried_input_len
                        else:
                            print("even after zeroshot the length of input is still (over limit) =", zero_shot_inpstr)
                            print("please fix this manually")
                            pdb.set_trace()
                    else:
                        total_input_toklen[task_name][split_name]+=final_total_len

                    for key in dp.keys():
                        if key not in onedp:
                            onedp[key] = dp[key]

                    out_dps.append(onedp)


            with jsonlines.open(f"{output_folder_root}/{task_name}/{split_name}.jsonl", "w") as w:
                for dp in out_dps:
                    w.write(dp)



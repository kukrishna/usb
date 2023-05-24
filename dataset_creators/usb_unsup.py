import json
import glob
import numpy as np
import jsonlines
import pdb
import os
import re
import argparse

import nltk
import difflib

parser = argparse.ArgumentParser()
parser.add_argument("--split-path", type=str, required=True)
parser.add_argument("--output-path", type=str, required=True)
parser.add_argument("--annot-path", type=str, required=True)
args = parser.parse_args()

split_path = args.split_path
OUTPUT_ROOT = args.output_path
annot_path = args.annot_path

np.random.seed(1729)

all_examples = {}

for split_name in ["train", "validation", "test"]:
    all_examples[split_name] = []
    suffixes = open(f"{split_path}/{split_name}.txt", "r").read().strip()
    if len(suffixes)==0:
        continue
    suffixes = suffixes.split("\n")

    fullpaths = [f"{annot_path}/{suf}" for suf in suffixes]


    for (path, suf) in zip(fullpaths, suffixes):
        obj = json.load(open(path, "r", encoding="utf-8"))
        obj["assignmentId"] = suf
        all_examples[split_name].append(obj)


test_examples = all_examples["test"]
val_examples = all_examples["validation"]
train_examples = all_examples["train"]



def showdiff(fr, to):
    differ = difflib.Differ()

    line = ""
    normal_spans = []
    deleted_spans = []
    added_spans = []

    for entry in differ.compare(nltk.word_tokenize(fr), nltk.word_tokenize(to)):
        if entry[0]=="+":
            text = entry[1:]
            start_idx = len(line)
            end_index = start_idx+len(text)
            added_spans.append((start_idx, end_index))
            line+=text
        elif entry[0]=="-":
            text = entry[1:]
            start_idx = len(line)
            end_index = start_idx+len(text)
            deleted_spans.append((start_idx, end_index))
            line+=text
        elif entry[0]=="?":
            continue
        else:
            text = entry[1:]
            start_idx = len(line)
            end_index = start_idx+len(text)
            normal_spans.append((start_idx, end_index))
            line+=text

    return {
        "line": line,
        "normal_spans": normal_spans,
        "deleted_spans": deleted_spans,
        "added_spans": added_spans
    }


os.makedirs(OUTPUT_ROOT, exist_ok=True)

DS_VARIANT = "unsupported_span_prediction"
os.mkdir(f"{OUTPUT_ROOT}/{DS_VARIANT}")

for (split_name, split_examples) in zip(["train", "validation", "test"],
                                        [train_examples, val_examples, test_examples]):
    clusterwise_dps = []



    for one_dp in split_examples:
        assignment_id = one_dp["assignmentId"]

        before_sents =  [x["txt"] for x in one_dp["annotations"]["orig_summary"]]
        after_sents = one_dp["annotations"]["annotations"]["summary_sents"]
        refs = one_dp["annotations"]["annotations"]["refs"]

        assert len(before_sents) == len(after_sents)

        for summ_idx,(before_sent,after_sent,ref_group) in enumerate(zip(before_sents, after_sents, refs)):
            if not after_sent:
                # ignoring all sentences which were completely deleted because i think the model will learn that as bias
                continue

            input_sents = [one_dp["annotations"]["source"][oneref]["txt"] for oneref in ref_group]

            if before_sent==after_sent:
                clusterwise_dps.append({
                    "summ_idx": summ_idx,
                    "id": assignment_id,
                    "input_lines": input_sents,
                    "before_lines": [before_sent],
                    "after_lines": [after_sent]
                })
            elif before_sent!=after_sent:
                diff_obj = showdiff(before_sent, after_sent)
                if len(diff_obj["added_spans"])>0:
                    continue  # ignore the things that have some addition from the annotator

                line = diff_obj["line"]
                normal_spans = diff_obj["normal_spans"]
                deleted_spans = diff_obj["deleted_spans"]

                new_sent = ""
                while len(normal_spans)>0 or len(deleted_spans)>0:
                    normal_charind = normal_spans[0][0] if len(normal_spans)>0 else 99999999
                    deleted_charind = deleted_spans[0][0] if len(deleted_spans)>0 else 99999999

                    if normal_charind<deleted_charind:
                        l,r = normal_spans[0]
                        new_sent += line[l:r]
                        normal_spans = normal_spans[1:]
                    else:
                        l, r = deleted_spans[0]
                        new_sent += "[]"+line[l:r]+"[/]"
                        deleted_spans = deleted_spans[1:]

                new_sent = new_sent.replace("[/][]","")   # merge contigous spans of deletions
                new_sent = new_sent.strip()

                clusterwise_dps.append({
                    "id": assignment_id,
                    "summ_idx": summ_idx,
                    "input_lines": input_sents,
                    "summary": before_sent,
                    "annotated_summary": new_sent
                })


    with jsonlines.open(f"{OUTPUT_ROOT}/{DS_VARIANT}/{split_name}.jsonl", "w") as w:
        for dp in clusterwise_dps:
            w.write(dp)

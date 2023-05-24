import json
import glob
import numpy as np
import jsonlines
import pdb
import os
import re
import argparse


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


os.makedirs(OUTPUT_ROOT, exist_ok=True)

DS_VARIANT = "evidence_extraction"
os.mkdir(f"{OUTPUT_ROOT}/{DS_VARIANT}")


for (split_name, split_examples) in zip(["train", "validation", "test"],
                                        [train_examples, val_examples, test_examples]):
    all_dps = []

    for one_dp in split_examples:
        assignment_id = one_dp["assignmentId"]

        source_sents = [x["txt"] for x in one_dp["annotations"]["source"]]
        after_sents = one_dp["annotations"]["annotations"]["summary_sents"]
        refs = one_dp["annotations"]["annotations"]["refs"]

        summary_lines = []
        all_refs = []

        for after_sent,ref_group in zip(after_sents, refs):
            if not after_sent:
                # ignoring all sentences which were completely deleted
                continue

            summary_lines.append(after_sent)
            all_refs.append(ref_group)

        all_dps.append({
            "id": assignment_id,
            "input_lines": source_sents,
            "summary_lines": summary_lines,
            "evidence_labels": all_refs
        })

    with jsonlines.open(f"{OUTPUT_ROOT}/{DS_VARIANT}/{split_name}.jsonl", "w") as w:
        for dp in all_dps:
            w.write(dp)

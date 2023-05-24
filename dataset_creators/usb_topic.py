import json
import glob
import numpy as np
import jsonlines
import pdb
import os
from collections import defaultdict
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

DS_VARIANT = "topicbased_summarization"
os.mkdir(f"{OUTPUT_ROOT}/{DS_VARIANT}")

for (split_name, split_examples) in zip(["train", "validation", "test"],
                                        [train_examples, val_examples, test_examples]):
    full_dps = []
    for one_dp in split_examples:
        assignment_id = one_dp["assignmentId"]

        source_lines = one_dp["annotations"]["source"]

        # remove the headers because otherwise they will give away the answer too easily
        # but retain the title of the article
        source_sents = [x["txt"] for x in source_lines if ((not x["is_header"]) or (x["section_index"]==0))]

        user_annots = one_dp["annotations"]["annotations"]

        summary_sents = user_annots["summary_sents"]
        refs = user_annots["refs"]

        sectionwise_summaries = defaultdict(list)

        for (sent, ref_group) in zip(summary_sents, refs):
            this_source_lines = [source_lines[x] for x in sorted(ref_group)]
            this_source_sents = [x["txt"] for x in this_source_lines]
            this_source_sections = [x["section_name"] for x in this_source_lines]

            if len(set(this_source_sections)) == 1:
                only_section = set(this_source_sections).pop()
                sectionwise_summaries[only_section].append(sent)

        for (_i,(section_name, sents)) in enumerate(sectionwise_summaries.items()):
            full_dps.append({
                "summ_idx": _i,
                "id": assignment_id,
                "input_lines": source_sents,
                "topic_name": section_name.rstrip(".").strip(),
                "output_lines": sents
            })

    with jsonlines.open(f"{OUTPUT_ROOT}/{DS_VARIANT}/{split_name}.jsonl", "w") as w:
        for dp in full_dps:
            w.write(dp)

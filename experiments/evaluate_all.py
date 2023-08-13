import pdb
import re
import os

import nlp
import argparse
import numpy as np
import jsonlines, json
from tqdm import tqdm
from collections import defaultdict
import glob
import pandas as pd

from sklearn import metrics

import difflib
from seqeval.metrics import classification_report as seqeval_classifcation_report
from nltk.tokenize import TreebankWordTokenizer as twt

def showdiff_helper(fr, to):
    differ = difflib.Differ()

    line = ""
    normal_spans = []
    deleted_spans = []
    added_spans = []

    for entry in differ.compare(fr, to):
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


def get_tagged_seq_helper(istring, ostring):
    s1 = istring
    s2 = ostring
    obj = showdiff_helper(s1,s2)

    deleted_spans = obj["deleted_spans"]
    added_spans = obj["added_spans"]
    normal_spans = obj["normal_spans"]
    line = obj["line"]

    open_positions = []
    close_positions = []

    raw_str = ""
    while len(normal_spans)>0 or len(deleted_spans)>0:
        normal_charind = normal_spans[0][0] if len(normal_spans)>0 else 99999999
        deleted_charind = deleted_spans[0][0] if len(deleted_spans)>0 else 99999999
        added_charind = added_spans[0][0] if len(added_spans)>0 else 99999999

        if normal_charind<deleted_charind and normal_charind<added_charind:
            l,r = normal_spans[0]
            normal_spans = normal_spans[1:]
            raw_str += line[l+1:r]
        elif deleted_charind<normal_charind and deleted_charind<added_charind:
            l, r = deleted_spans[0]
            deleted_spans = deleted_spans[1:]
            raw_str += line[l+1:r]
        elif added_charind<normal_charind and added_charind<deleted_charind:
            l, r = added_spans[0]
            if line[l:l+4]==' [ ]':
                if len(open_positions)==len(close_positions):
                    # should close
                    open_positions.append(len(raw_str))
            if line[l:l+6]==' [ / ]':
                if len(open_positions)==len(close_positions)+1:
                    close_positions.append(len(raw_str))
            added_spans = added_spans[1:]

    if len(open_positions)==len(close_positions)+1:
        close_positions.append(len(raw_str))

    highlight_spans = list(zip(open_positions, close_positions))

    words = []
    tags = []
    wordspans = list(twt().span_tokenize(istring))

    for onespan in wordspans:
        l, r = onespan
        words.append(istring[l:r])
        should_mark = False
        for (left_inc, right_ex) in highlight_spans:
            if l>=left_inc and l<right_ex:
                should_mark=True

        if should_mark:
            tags.append('I-MISC')
        else:
            tags.append('O')

    if tags[0]=='I-MISC':
        tags[0]='B-MISC'

    for j in range(1,len(tags)):
        if tags[j]=='I-MISC' and tags[j-1]=='O':
            tags[j]='B-MISC'

    return words, tags


def evaluate_spanoverlap(test_preds):
    y_true = []
    y_pred = []

    for onepred in test_preds:
        orig_str=onepred["base_string"]
        pred_str=onepred["prediction"]
        gt_str=onepred["ground_truth"]

        gt_words, gt_tags = get_tagged_seq_helper(istring=orig_str,
                                                  ostring=gt_str)
        pred_words, pred_tags = get_tagged_seq_helper(istring=orig_str,
                                                      ostring=pred_str)
        assert gt_words==pred_words
        y_pred.append(pred_tags)
        y_true.append(gt_tags)

    results = seqeval_classifcation_report(y_true, y_pred, output_dict=True)

    to_return = {
        "precision": results["MISC"]["precision"],
        "recall": results["MISC"]["recall"],
        "f1": results["MISC"]["f1-score"]
    }

    return to_return


def evaluate_rouge_using_huggingface(test_preds):
    rouge = nlp.load_metric('rouge')
    all_scores = defaultdict(list)
    for elem in test_preds:
        generated = elem["prediction"].lower()
        reference = elem["ground_truth"].lower()

    #     second one is ground truth
        rouge.add( prediction=generated, reference=reference)
        all_scores["exactmatch"].append(generated==reference)


    score = rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])

    all_scores["rouge1"].append( score['rouge1'].mid.fmeasure )
    all_scores["rouge2"].append( score['rouge2'].mid.fmeasure )
    all_scores["rougel"].append( score['rougeL'].mid.fmeasure )
    for (k,v) in all_scores.items():
        all_scores[k]=np.mean(v)

    return all_scores



def calc_metrics(y_true, y_pred_class, y_pred_cont, label_names):
    return_dict={}

    num_points=y_true.shape[0]
    num_classes=y_true.shape[1]

    assert num_classes==1
    y_true = y_true[:,0]
    y_pred_class = y_pred_class[:,0]
    y_pred_cont = y_pred_cont[:,0]

    return_dict["precision"]=metrics.precision_score(y_true, y_pred_class)
    return_dict["recall"]=metrics.recall_score(y_true, y_pred_class)
    return_dict["f1"]=metrics.f1_score(y_true, y_pred_class)
    return_dict["accuracy"]=metrics.accuracy_score(y_true, y_pred_class)
    return_dict["auc"]=metrics.roc_auc_score(y_true, y_pred_cont)

    return return_dict


def calc_main(prediction_glob):
    all_pred_files = glob.glob(prediction_glob)
    print("processing files...", all_pred_files)

    pred_dps = []

    for fpath in all_pred_files:
        this_dps = list(jsonlines.open(fpath))
        pred_dps.extend(this_dps)

    tasks = [x["task"] for x in pred_dps]
    unique_tasks = set(tasks)


    all_task_metrics = {}

    for task_name in unique_tasks:
        evaluation_dps = []
        if task_name in ["multisentence_compression", "fixing_factuality", "topicbased_summarization", "abstractive_summarization"]:
            for (this_task, pred) in zip(tasks, pred_dps):
                if this_task==task_name:
                    if "prediction" not in pred:
                        continue
                    evaluation_dps.append({"prediction": pred["prediction"], "ground_truth": pred["output_string"]})

            rouge_metrics=evaluate_rouge_using_huggingface(evaluation_dps)
            all_task_metrics[task_name]=rouge_metrics

        if task_name in ["unsupported_span_prediction"]:
            for (this_task, pred) in zip(tasks, pred_dps):
                if this_task==task_name:
                    if "prediction" not in pred:
                        continue
                    if "[]" in pred["prediction"]:
                        evaluation_dps.append({"base_string": pred["summary"], "prediction": pred["prediction"], "ground_truth": pred["annotated_summary"]})
                    else:
                        # if the model did not generate any span then take its output to be the same as input summary
                        # avoids unncessary penalization of model for not copying the input summary properly
                        evaluation_dps.append({"base_string": pred["summary"], "prediction": pred["summary"], "ground_truth": pred["annotated_summary"]})
            overlap_metrics = evaluate_spanoverlap(evaluation_dps)
            all_task_metrics[task_name]=overlap_metrics

        if task_name in ["extractive_summarization", "factuality_classification", "evidence_extraction"]:
            all_labels = []
            all_probs = []
            for (this_task, pred) in zip(tasks, pred_dps):
                if "prediction" not in pred:
                    continue
                if this_task==task_name:
                    if task_name=="factuality_classification":
                        all_labels.append(pred["label"])

                        if len(pred["prediction"])!=1:
                            print("found one offending entry")
                            all_probs.append(0)  # default value since most summaries should be factual in real world
                        else:
                            # 1 minus because here Yes means factual, prediction contains yes_prob
                            all_probs.append(1-pred["prediction"][0])

                    elif task_name=="extractive_summarization":
                        if len(pred["labels"])!=len(pred["prediction"]):
                            print("found one offending entry")
                            pred["prediction"] = pred["prediction"] [:len(pred["labels"])]  # truncate
                            pred["prediction"] = pred["prediction"] + \
                                                 [0 for _ in range(len(pred["labels"])-len(pred["prediction"]))]
                                                # again default to 0 because most things not important

                        all_labels.extend(pred["labels"])
                        all_probs.extend(pred["prediction"])

                    elif task_name=="evidence_extraction":
                        input_lines = pred["input_lines"]
                        if len(pred["input_lines"])!=len(pred["prediction"]):
                            print("found one offending entry")
                            pred["prediction"] = pred["prediction"][:len(pred["input_lines"])]  # truncate
                            pred["prediction"] = pred["prediction"] + \
                                                 [0 for _ in range(len(pred["input_lines"])-len(pred["prediction"]))]
                                                # default is 0 because most things wont be evidence for a given sentence

                        onehot_labels = [0 for _ in input_lines]
                        for idx in pred["labels"]:
                            onehot_labels[idx]=1

                        all_labels.extend(onehot_labels)
                        all_probs.extend(pred["prediction"])

            all_pred_class = np.round(all_probs).astype(int)

            all_labels = [[x] for x in all_labels]
            all_probs = [[x] for x in all_probs]
            all_pred_class = [[x] for x in all_pred_class]

            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)
            all_pred_class = np.array(all_pred_class)

            classification_metrics = calc_metrics(y_true=all_labels,
                                                  y_pred_cont=all_probs,
                                                  y_pred_class=all_pred_class,
                                                  label_names=["positive"])

            all_task_metrics[task_name]=classification_metrics

    return all_task_metrics


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='calculate rouge scores from output file')
    parser.add_argument(
        '--prediction-file',
        dest='pred_file',
        help='Prediction file',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--output-file',
        dest='out_file',
        help='Output file',
        type=str,
        required=True,
    )

    args = parser.parse_args()
    pred_file = args.pred_file
    out_file = args.out_file

    all_task_metrics = calc_main(prediction_glob=pred_file)

    df = pd.DataFrame(all_task_metrics)

    print(df)

    json.dump(all_task_metrics, open(out_file, "w"))



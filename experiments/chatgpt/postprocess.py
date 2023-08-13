import pdb

import jsonlines
import argparse
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string
stop_words = set(stopwords.words('english'))
import re



def postprocess_cluster2sent(dp):
    return dp['prediction']

def postprocess_full2full(dp):
    return dp['prediction']

def postprocess_fixfactuality(dp):
    return dp['prediction']

def postprocess_unsup_spans(dp):
    return dp['prediction']

def postprocess_full2section(dp):
    return dp['prediction']

def postprocess_factuality_classification(dp):
    text = dp['prediction'].lower()
    words = word_tokenize(text)
    words = [each.lower().strip(string.punctuation) for each in words]

    # ATTENTION: return [1.0] for FACTUAL and [0.0] for NONFACTUAL
    text = ' '.join(words).strip()
    # default response is a pretty subjective opinion since in the dataset it's 50-50% chance for each label.
    # but since in real world most summaries (even model-generated) are factual, we set that as default
    default_response = [1.0]

    if 'summary is not consistent' in text:
        return [0.0]
    elif 'summary is partially consistent' in text:
        return [0.0]
    elif 'summary is mostly consistent' in text:
        return [0.0]
    elif 'some' in words and 'claims' in words and 'supported' in words:
        return [0.0]

    elif 'summary is consistent' in text:
            return [1.0]
    elif 'partial' in text or 'mostly' in text:
            return [0.0]
    elif 'insufficient evidence' in text:
            return [0.0]
    elif 'are supported' in text:
            return [1.0]
    elif 'factually accurate' in text:
        return [1.0]
    elif text == 'not supported':
        return [0.0]
    elif text == 'supported':
        return [1.0]
    elif 'yes' in words or 'true' in words :
        return [1.0]
    elif 'no' in words or 'false' in words:
        return [0.0]
    else:
        return default_response

def postprocess_importance_classification(dp):
    inp_string = dp['input_string']
    pred_string = dp['prediction']
    all_src_content = inp_string.split('CONTENT:')[-1].strip()

    sent_idx = re.findall('SENT\d+', all_src_content)
    sent_idx_labels = ['no'] * len(sent_idx)
    for l in pred_string.strip().split(' '):
        l = l.strip()
        if l in sent_idx:
            l_id = sent_idx.index(l)
            sent_idx_labels[l_id] = 'yes'

    sent_preds = [1.0 if each == 'yes' else 0.0 for each in sent_idx_labels]
    return sent_preds


def postprocess_evidence_extraction(dp):
    inp_string = dp['input_string']
    pred_string = dp['prediction']
    all_src_content = inp_string.split('CONTENT:')[-1].strip()

    sent_idx = re.findall('SENT\d+', all_src_content)
    sent_idx_labels = ['no'] * len(sent_idx)
    for l in pred_string.strip().split(' '):
        l = l.strip()
        if l in sent_idx:
            l_id = sent_idx.index(l)
            sent_idx_labels[l_id] = 'yes'

    sent_preds = [1.0 if each == 'yes' else 0.0 for each in sent_idx_labels]
    return sent_preds



ALL_POSTPROC_FUNCS = {
"multisentence_compression": postprocess_cluster2sent,
"abstractive_summarization": postprocess_full2full,
"fixing_factuality": postprocess_fixfactuality,
"unsupported_span_prediction": postprocess_unsup_spans,
"topicbased_summarization": postprocess_full2section,
"factuality_classification": postprocess_factuality_classification,
"extractive_summarization": postprocess_importance_classification,
"evidence_extraction": postprocess_evidence_extraction,
}

if __name__=="__main__":

    np.random.seed(1337)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    input_dps = list(jsonlines.open(input_file))

    print("TASK NAME = ", input_dps[0]["task"])

    with jsonlines.open(output_file, "w") as w:
        for dp in tqdm(input_dps):
            task_name = dp["task"]
            postproc_func = ALL_POSTPROC_FUNCS[task_name]
            final_prediction = postproc_func(dp)
            dp["prediction"] = final_prediction
            w.write(dp)


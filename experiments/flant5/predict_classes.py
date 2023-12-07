import pdb

import argparse
import torch.nn.functional
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
import numpy as np
import jsonlines
from copy import deepcopy


def predict_classification(dp, model, tokenizer, nbeams, max_input_len, max_decode_len):
    yes_id = tokenizer.convert_tokens_to_ids("▁Yes")
    no_id = tokenizer.convert_tokens_to_ids("▁No")

    inputs = tokenizer(dp["input_string"], return_tensors="pt", truncation=True, max_length=max_input_len)
    input_ids = inputs.input_ids.cuda()


    gen_output = model.generate(input_ids,
                                return_dict_in_generate=True,
                                output_scores=True,
                                max_length=max_decode_len,          # have to set again :( cant read from saved model
                                num_beams=nbeams)
    gen_tokids = gen_output["sequences"][0]
    gen_tokids = gen_tokids[1:]  # throw away the BOS token (0 in this case)
    gen_logits = gen_output["scores"]

    if "beam_indices" in gen_output:
        gen_beam_indices = gen_output["beam_indices"][0]
        gen_beam_indices = [x for x in gen_beam_indices if x>=0]    # ignore -1
        gen_logits = [gen_logits[step][b_idx:b_idx+1] for (step, b_idx) in enumerate(gen_beam_indices)]

    all_yes_probs = []

    for (this_tokid, this_logits) in zip(gen_tokids, gen_logits):
        if this_tokid in [yes_id, no_id]:
            this_logits = this_logits.squeeze(0)
            probs = torch.nn.functional.softmax(this_logits.type(torch.float32), dim=0)  # can't softmax a float16 arr

            yes_prob = probs[yes_id].item()
            no_prob = probs[no_id].item()
            yes_prob_renorm = yes_prob/(yes_prob+no_prob)
            all_yes_probs.append(yes_prob_renorm)

    dp["prediction"] = all_yes_probs



if __name__=="__main__":

    np.random.seed(1337)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--nbeams", type=int, default=1)
    parser.add_argument("--max-input-len", type=int, default=8192)
    parser.add_argument("--max-decode-len", type=int, default=768)
    args = parser.parse_args()

    input_file = args.input_file
    model_path = args.model_path
    output_file = args.output_file
    task = args.task
    nbeams = args.nbeams
    max_input_len = args.max_input_len
    max_decode_len = args.max_decode_len

    model = T5ForConditionalGeneration.from_pretrained(model_path).cuda()
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    all_dps = list(jsonlines.open(input_file))

    for dp in tqdm(all_dps):
        predict_classification(dp, model, tokenizer, nbeams, max_input_len, max_decode_len)


    with jsonlines.open(output_file, "w") as w:
        for dp in all_dps:
            w.write(dp)
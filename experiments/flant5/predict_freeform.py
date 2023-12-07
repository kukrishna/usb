import pdb

import argparse
import torch.nn.functional
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import jsonlines


def predict_generation(dp, model: T5ForConditionalGeneration, tokenizer, nbeams, max_input_len, max_decode_len):
    inputs = tokenizer(dp["input_string"], return_tensors="pt", truncation=True, max_length=max_input_len)
    input_ids = inputs.input_ids.cuda()

    decoder_input_ids = None

    if "generation_seed" in dp:
        generation_seed = dp["generation_seed"]
        if len(generation_seed)>0:
            decoder_input_ids = tokenizer.encode(generation_seed, add_special_tokens=False)
            decoder_input_ids = [0]+decoder_input_ids
            decoder_input_ids = torch.tensor([decoder_input_ids]).cuda()


    if decoder_input_ids!=None:
        gen_output = model.generate(input_ids,
                                    return_dict_in_generate=True,
                                    decoder_input_ids=decoder_input_ids,
                                    output_scores=False,
                                    max_length=max_decode_len,          # have to set again :( cant read from saved model
                                    num_beams=nbeams)
    else:
        gen_output = model.generate(input_ids,
                                    return_dict_in_generate=True,
                                    output_scores=False,
                                    max_length=max_decode_len,          # have to set again :( cant read from saved model
                                    num_beams=nbeams)

    gen_tokids = gen_output["sequences"][0]

    gen_tokids = gen_tokids[1:] # first token is pad
    if gen_tokids[-1].item()==tokenizer.eos_token_id:
        gen_tokids = gen_tokids[:-1]

    gen_string = tokenizer.decode(gen_tokids)
    print(gen_string)
    dp["prediction"] = gen_string




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
        predict_generation(dp, model, tokenizer, nbeams, max_input_len, max_decode_len)


    with jsonlines.open(output_file, "w") as w:
        for dp in all_dps:
            w.write(dp)
import os
import sys
import json
import torch
import argparse
from tqdm import tqdm

from modeling_t5 import T5ForConditionalGeneration
from transformers import T5Tokenizer
from GeDi_t5 import GeDi

device = "cuda" if torch.cuda.is_available() else "cpu"

def gedi_decode(model, tokenizer, gedi, gedi_params, input_text):
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids

    if device == "cuda":
        input_ids = input_ids.to(device)

    outputs = model.generate(input_ids, max_length=256, tokenizer=tokenizer, gedi_decoder=gedi, gedi_params=gedi_params)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--top_beam", type=int, default=16, help="beam size for generation candidates to be reranked")
    parser.add_argument("-i", "--ignore_first_t", type=int, default=3, help="GeDi discriminator ignores first i-steps during generation.")
    parser.add_argument("-l", "--gedi_lambda", type=float, default=0.8, help="GeDi lambda weight (between 0 and 1).")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Print verbose outputs")
    parser.add_argument("-m", "--t5_model_size",  default="t5-large", help="T5 model size")
    parser.add_argument("-g", "--generator_model_path",  default=None, help="generator model path")
    parser.add_argument("-d", "--discriminator_model_path",  default=None, help="gedi discriminator model path")
    args = parser.parse_args()
    print(args)


    # GeDi parameters (TODO: argparse)
    gedi_params = {}
    gedi_params["ignore_first_t"] = args.ignore_first_t
    gedi_params["top_beam"] = args.top_beam
    gedi_params["gedi_lambda"] = args.gedi_lambda
    gedi_params["verbose"] = args.verbose

    t5_model_size = args.t5_model_size
    seq2seq_model_path = args.generator_model_path
    gedi_dscrm_path = args.discriminator_model_path


    tokenizer = T5Tokenizer.from_pretrained(t5_model_size)

    # seq2seq generator (for RORStories)
    model = T5ForConditionalGeneration.from_pretrained(seq2seq_model_path)

    # GeDi discriminator
    gedi = GeDi(gedi_dscrm_path)

    if device == "cuda":
        model = model.to(device)
        gedi.model = gedi.model.to(device)


    ### example usage (TO BE EDITED) ###
    dataset="moralstories"
    dataset="rocstories"
    sp="dev" #"test"

    f_in_path = f"./human_eval/{dataset}_{sp}.json"
    if gedi_params["gedi_lambda"] == 0.0:
        #f_out = open(f"./human_eval/pred_{dataset}_{sp}_no_gedi.txt", 'w')
        f_out = open(f"./tmp/pred_{dataset}_{sp}_no_gedi.txt", 'w')
    else:
        g = gedi_params["gedi_lambda"]
        #f_out = open(f"./human_eval/pred_{dataset}_{sp}_{g}.txt", 'w')
        f_out = open(f"./tmp/pred_{dataset}_{sp}_{g}.txt", 'w')

    num_lines = sum(1 for line in open(f_in_path, 'r'))
    for row in tqdm(open(f_in_path, 'r'), total=num_lines, mininterval=1):
        row = json.loads(row)
        input_text = row["text"]
        oracle_text = row["summary"]
        outputs = gedi_decode(model, tokenizer, gedi, gedi_params, input_text)
        pred_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(pred_text[0])
        f_out.write(pred_text[0]+'\n')


if __name__ == "__main__":
    main()

import os
import sys
import json
import torch
from tqdm import tqdm

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import LogitsProcessorList, MinLengthLogitsProcessor
from torch import nn, argmax, multinomial
from GeDi_t5 import GeDi

device = "cuda" if torch.cuda.is_available() else "cpu"

def gedi_decode(model, tokenizer, gedi, gedi_params, input_text):
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids

    if device == "cuda":
        input_ids = input_ids.to(device)

    outputs = model.generate(input_ids, max_length=256, tokenizer=tokenizer, gedi_decoder=gedi, gedi_params=gedi_params)
    return outputs

# GeDi parameters (TODO: argparse)
gedi_params = {}
gedi_params["ignore_first_t"] = 3
gedi_params["top_beam"] = 16
gedi_params["gedi_lambda"] = 0.8
gedi_params["verbose"] = False

t5_model_size = "t5-large"
seq2seq_model_path = "./roc_models/"
gedi_dscrm_path = "./cls_models/"


tokenizer = T5Tokenizer.from_pretrained(t5_model_size)

# seq2seq generator (for RORStories)
model = T5ForConditionalGeneration.from_pretrained(seq2seq_model_path)

# GeDi discriminator
gedi = GeDi(gedi_dscrm_path)

if device == "cuda":
    model = model.to(device)
    gedi.model = gedi.model.to(device)


### main ###
dataset="moralstories"
dataset="rocstories"

f_in_path = f"./human_eval/{dataset}_test.json"
f_out = open(f"./human_eval/pred_{dataset}_test.txt", 'w')
num_lines = sum(1 for line in open(f_in_path, 'r'))
for row in tqdm(open(f_in_path, 'r'), total=num_lines, mininterval=1):
    row = json.loads(row)
    input_text = row["text"]
    oracle_text = row["summary"]
    outputs = gedi_decode(model, tokenizer, gedi, gedi_params, input_text)
    pred_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(pred_text[0])
    f_out.write(pred_text[0]+'\n')


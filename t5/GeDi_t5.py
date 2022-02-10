import os
import sys
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

class GeDi:
    def __init__(self, model_name_or_path):
        print("GeDi model loading ... ...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        print("GeDi model loading ... ... done.")

    def predict(self, input_texts):
        input_text = f"{input_texts}"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        if device == "cuda":
            input_ids = input_ids.to(device)

        outputs = self.model.generate(input_ids, num_beams=4, return_dict_in_generate=True, num_return_sequences=4, output_scores=True)
        seq = outputs["sequences"]
        outputs_text = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
        scores = outputs["sequences_scores"]
        best_output = outputs_text[0]
        #best_prob = torch.exp(scores[0]) # prob
        best_prob = scores[0] # log-prob
        return best_output, best_prob

    def predict_from_tokenized(self, input_ids):
        outputs = self.model.generate(input_ids, num_beams=4, return_dict_in_generate=True, num_return_sequences=4, output_scores=True)
        seq = outputs["sequences"]
        outputs_text = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
        scores = outputs["sequences_scores"]
        print(outputs_text)
        print(scores)
        best_output = outputs_text[0]
        #best_prob = torch.exp(scores[0]) # prob
        best_prob = scores[0] # log-prob
        return best_output, best_prob


    def get_loss(self, input_ids, labels):
        with torch.no_grad():

            if device == "cuda":
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                self.model = self.model.to(device)

            #TODO: in future, we want to get losses without reduction  
            # c.f. https://github.com/huggingface/transformers/issues/11988
            #outputs = self.model(input_ids=input_ids, labels=labels, reduction='none')

            outputs = self.model(input_ids=input_ids, labels=labels)
            return outputs["loss"]


# unit_test
def main_gedi(model_path):
    gedi = GeDi(model_path)
    text = "It is trivial to understand what has gone wrong with a project to know how to fix it."
    best_output, best_prob = gedi.predict(text)
    print(best_output)
    print(best_prob)


# unit_test
if __name__ == "__main__":
    main_gedi("./cls_models/")

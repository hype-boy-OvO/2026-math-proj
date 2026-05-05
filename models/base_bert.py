from transformers import BertModel, BertTokenizer
import torch
import os

class BertTiny:
    def __init__(self, model_name="google-bert/bert-base-uncased", device=None):
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # hook 설정
        self.activations = {}
        def hook_fn(module, input, output):  
            self.activations["ffn"] = output[:,0,:].clone().cpu()
        self.model.encoder.layer[-1].intermediate.register_forward_hook(hook_fn)


    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]
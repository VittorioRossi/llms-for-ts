import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
from transformers import BigBirdPegasusForConditionalGeneration, PegasusTokenizer
from transformers import PegasusForConditionalGeneration
from abc import ABC, abstractmethod
import os
import numpy as np


class LLM(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, batch: list[str]) -> str:
        pass


class HuggingFaceLLM(LLM):
    def __init__(self, model: str, example_output="00.0", target_size=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = self.setup_generator(model, example_output, target_size)

    def setup_generator(self, model_name, example_output="00.0", target_size=1):
        token = os.environ.get("HUGGINGFACE_TOKEN")
        model = self.load_model(model_name, token)
        tokenizer = self.load_tokenizer(model_name, token)

        tokenizer = set_pad_token_if_missing(tokenizer)
        max_new_tok = compute_new_tokens(target_size, example_output, tokenizer)

        def gen(texts, **kwargs):
            inputs = self.tokenize_inputs(tokenizer, texts)
            outputs = self.generate_outputs(model, tokenizer, inputs, max_new_tok)
            results = self.decode_outputs(tokenizer, texts, outputs, target_size=target_size)
            return results

        return gen
    
    def generate(self, batch):
        return self.generator(batch)

    def load_model(self, model_name, token):
        model_kwargs = {}
        if 'bert' in model_name.lower():
            model_kwargs['is_decoder'] = True
        
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir="models",
            torch_dtype="auto",
            token=token,
            **model_kwargs
        ).to(self.device)

    def load_tokenizer(self, model_name, token):
        tokenizer_kwargs = {}
        if 'bert' in model_name.lower():
            tokenizer_kwargs['padding_side'] = 'left'
        return AutoTokenizer.from_pretrained(model_name, token=token, **tokenizer_kwargs)

    def tokenize_inputs(self, tokenizer, texts):
        return tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

    def generate_outputs(self, model, tokenizer, inputs, max_new_tokens):
        model.to(self.device)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id
        )

    def decode_outputs(self, tokenizer, texts, outputs, target_size):
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        torch.cuda.empty_cache()

        results = []
        for text, generated_text in zip(texts, generated_texts):

            if text in generated_text:
                generated_text = generated_text[len(text):]
            
            preds = clean_pred(generated_text, target_size)
            if np.isnan(preds).any():
                print(f"Failed to convert prediction '{generated_text}' to float")

            results.append(preds)


        return results

def set_pad_token_if_missing(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def compute_new_tokens(target_size, example_output, tokenizer):
    example_tokens = tokenizer(example_output, add_special_tokens=False)['input_ids']
    return target_size * len(example_tokens)

def clean_pred(pred: str, target_size: int):
    # Split the predicted string into tokens
    tokens = pred.strip().split()[:target_size]

    # Initialize the result list
    res = []

    for token in tokens:
        try:
            # Attempt to convert each token to a float
            res.append(float(token))
        except ValueError:
            # If conversion fails, append np.nan
            res.append(np.nan)

    # If the number of tokens is less than target_size, append np.nan to the result
    while len(res) < target_size:
        res.append(np.nan)

    return res
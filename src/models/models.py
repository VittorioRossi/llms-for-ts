import torch
from transformers import AutoModel, AutoTokenizer, BertLMHeadModel
from abc import ABC, abstractmethod
import os
import numpy as np


def set_pad_token_if_missing(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def compute_new_tokens(target_size, example_output, tokenizer):
    # Tokenize the original input text to find its length
    original_input_ids = tokenizer(example_output, add_special_tokens=False)['input_ids']
    original_input_length = len(original_input_ids)

    return original_input_length*target_size

def clean_pred(pred:str, target_size:int):
    # Transform the predicted tokens in a float
    pred = pred.split(" ")[:target_size]
    res = []
    for el in pred:
        try:
            res.append(float(el))
        except:
            res.append(np.nan)

        
    return res


class LLM(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
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

        def gen(texts, max_new_tokens=max_new_tok):
            inputs = self.tokenize_inputs(tokenizer, texts)
            outputs = self.generate_outputs(model, tokenizer, inputs, max_new_tokens)
            results = self.decode_outputs(tokenizer, texts, outputs, target_size=target_size)
            return results

        return gen
    
    def generate(self, batch):
        return self.generator(batch)[0]

    def load_model(self, model_name, token):
        # Load the model with appropriate class
        if 'bert' in model_name.lower():
            from transformers import BertModel
            return BertLMHeadModel.from_pretrained(
                model_name,
                cache_dir="models",
                torch_dtype="auto",
                use_auth_token=token

            ).to(self.device)
        else:
            return AutoModel.from_pretrained(
                model_name,
                cache_dir="models",
                torch_dtype="auto",
                use_auth_token=token
            ).to(self.device)

    def load_tokenizer(self, model_name, token):
        tokenizer_kwargs = {}
        if 'bert' in model_name.lower():
            tokenizer_kwargs['padding_side'] = 'left'
        return AutoTokenizer.from_pretrained(model_name, use_auth_token=token, **tokenizer_kwargs)

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

        original_lengths = [len(tokenizer(text, add_special_tokens=False)['input_ids']) for text in texts]
        new_texts = [generated_text[len(text):] for generated_text, text in zip(generated_texts, texts)]

        predictions = [
            tokenizer.decode(
                tokenizer(new_text, add_special_tokens=False)['input_ids'][original_length:],
                skip_special_tokens=True
            )
            for new_text, original_length in zip(new_texts, original_lengths)
        ]

        return [clean_pred(prediction, target_size) for prediction in predictions]

def set_pad_token_if_missing(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def compute_new_tokens(target_size, example_output, tokenizer):
    example_tokens = tokenizer(example_output, add_special_tokens=False)['input_ids']
    return target_size * len(example_tokens)

def clean_pred(pred: str, target_size: int):
    # Split the predicted tokens and take only the first `target_size` elements
    tokens = pred.split(" ")[:target_size]
    
    # Convert tokens to floats, using np.nan for any conversion errors
    return [float(token) if token.replace('.', '', 1).isdigit() else np.nan for token in tokens]
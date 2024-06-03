import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod
import os
import numpy as np

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




class LLM(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, batch: list[str]) -> str:
        pass


class HuggingFaceLLM(LLM):
    def __init__(self, model: str, example_output="00.0", target_size=1, max_token_mutliplier=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_token_mutliplier = max_token_mutliplier
        self.generator = self.setup_generator(model, example_output, target_size)

    def setup_generator(self, model_name, example_output="00.0", target_size=1):
        token = os.environ.get("HUGGINGFACE_TOKEN")
        model = self.load_model(model_name, token)
        tokenizer = self.load_tokenizer(model_name, token)

        tokenizer = set_pad_token_if_missing(tokenizer)
        max_new_tok = compute_new_tokens(target_size, example_output, tokenizer) * self.max_token_mutliplier

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
        tokenizer_kwargs['padding_side'] = 'left'
        return AutoTokenizer.from_pretrained(model_name, token=token, **tokenizer_kwargs)

    def tokenize_inputs(self, tokenizer, texts):
        return tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
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


class HuggingFaceLLMChat(HuggingFaceLLM):
    def __init__(self, model: str, example_output="00.0", target_size=1, max_token_mutliplier=1):
        super().__init__(model, example_output, target_size, max_token_mutliplier)
        self.max_token_mutliplier = max_token_mutliplier

    def setup_generator(self, model_name, example_output="00.0", target_size=1):
        token = os.environ.get("HUGGINGFACE_TOKEN")
        model = self.load_model(model_name, token)
        tokenizer = self.load_tokenizer(model_name, token)

        tokenizer = set_pad_token_if_missing(tokenizer)
        max_new_tok = compute_new_tokens(target_size, example_output, tokenizer) * self.max_token_mutliplier

        def gen(batch_messages, **kwargs):
            system_message = kwargs.get('system_message', "you are a time series forecasting model")
            preproces_batch = self.apply_system_message(batch_messages, system_message=system_message)
            inputs_batch = self.tokenize_batch(tokenizer, preproces_batch)
            outputs = self.generate_outputs(model, tokenizer, inputs_batch, max_new_tok)
            results = self.decode_outputs(tokenizer, preproces_batch, outputs, target_size=target_size)
            return results

        return gen
    
    
    def apply_system_message(self, batch_messages, system_message = "you are a time series forecasting model"):
        return [
            [
                {'role':'system', 'content':system_message},
                {'role':'user', 'content':message}
            ]
            for message in batch_messages
        ]

    def tokenize_batch(self, tokenizer, batch_messages):
        def apply_chat_template(messages):
            return tokenizer(
                tokenizer.eos_token.join([msg['content'] for msg in messages]),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )

        inputs_batch = [apply_chat_template(messages) for messages in batch_messages]


        padded = all(inputs_batch[0]['input_ids'].shape[1] == inputs['input_ids'].shape[1] for inputs in inputs_batch)
        
        if not padded:
            max_length = max(inputs['input_ids'].shape[1] for inputs in inputs_batch)  # Get the max length in the batch


            # Ensure all inputs are padded to the max length on the left
            for inputs in inputs_batch:
                padding_length = max_length - inputs['input_ids'].shape[1]
                inputs['input_ids'] = torch.nn.functional.pad(inputs['input_ids'], (padding_length, 0), value=tokenizer.pad_token_id)
                inputs['attention_mask'] = torch.nn.functional.pad(inputs['attention_mask'], (padding_length, 0), value=0)

        return {key: torch.cat([inputs[key] for inputs in inputs_batch], dim=0).to(self.device) for key in inputs_batch[0]}

    def decode_outputs(self, tokenizer, batch_messages, outputs, target_size):
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        torch.cuda.empty_cache()

        results = []
        for messages, generated_text in zip(batch_messages, generated_texts):
            original_text = ' '.join([msg['content'] for msg in messages])
            generated_text = generated_text[len(original_text):].strip()
            preds = clean_pred(generated_text, target_size)
            if np.isnan(preds).any():
                print(f"Failed to convert prediction '{generated_text}' to float")

            results.append(preds)

        return results
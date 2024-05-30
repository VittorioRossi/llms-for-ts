import torch
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer
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
    def __init__(self, model: str):
        # Detect if CUDA is available and set the device accordingly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.generator = self.setup_generator(model)
        except Exception as e:
            # Setup the pipeline if no exceptions
            self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)

    def setup_generator(self, model_name, target_size=1, example_output="00.0"):
        token = os.environ.get("HUGGINGFACE_TOKEN")
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     cache_dir="models",
                                                     torch_dtype="auto",
                                                     use_auth_token=token).to(self.device)
        tokenizer_kwargs = {}
        if model_name.__contains__('bert'):
            tokenizer_kwargs['padding_side'] = 'left'
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token, **tokenizer_kwargs)

        tokenizer = set_pad_token_if_missing(tokenizer)
        
        max_new_tok = compute_new_tokens(target_size, example_output, tokenizer)

        def gen(texts, max_new_tokens=None):
            # Ensure the tokenizer has a pad token
            if max_new_tokens is None:
                max_new_tokens = max_new_tok
            
            # Tokenize the input texts with padding and truncation enabled
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True)
            
            # Move the inputs to the appropriate device (CUDA or CPU)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate new tokens using the model
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id  # Ensure proper padding if needed
            )
            
            # Decode the generated tokens to text, skipping special tokens
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Clean up by deleting unnecessary variables and clearing cache
            del inputs
            del outputs
            torch.cuda.empty_cache()
            
            results = []
            for text, generated_text in zip(texts, generated_texts):
                # Tokenize the original input text to find its length
                original_input_ids = tokenizer(text, add_special_tokens=False)['input_ids']
                original_input_length = len(original_input_ids)

                # Tokenize the full generated text to find the portion that corresponds to the prediction
                full_generated_ids = tokenizer(generated_text, add_special_tokens=False)['input_ids']

                # Get the part of the generated text that corresponds to the prediction
                new_token_ids = full_generated_ids[original_input_length:]

                # Decode the new tokens to get the prediction text
                prediction = tokenizer.decode(new_token_ids, skip_special_tokens=True)

                preds = clean_pred(prediction, target_size)
                results.append(preds)

            return results
    
        return gen 

    def generate(self, prompt: str, max_new_tokens=None) -> str:
        if hasattr(self, 'generator'):
            return self.generator(prompt, max_new_tokens=max_new_tokens)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
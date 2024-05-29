import torch
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod
import os

def set_pad_token_if_missing(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

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

    def setup_generator(self, model_name):
        token = os.environ.get("HUGGINGFACE_TOKEN")
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     cache_dir="models",
                                                     torch_dtype="auto",
                                                     use_auth_token=token).to(self.device)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)

        tokenizer = set_pad_token_if_missing(tokenizer)

        def gen(texts, max_new_tokens=100):
            # Ensure the tokenizer has a pad token
            
            # Tokenize the input texts with padding and truncation enabled
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512  # Adjust as needed
            )
            
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
                results.append(prediction)

            return results
    
        return gen 

    def generate(self, prompt: str) -> str:
        if hasattr(self, 'generator'):
            return self.generator(prompt, max_new_tokens=100)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=100)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
import torch
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod
import os
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
            # Load the model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(model, cache_dir="models", torch_dtype=torch.float32).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        except Exception as e:
            # If there's an issue with loading from pretrained, setup a generator
            self.generator = self.setup_generator(model)
        else:
            # Setup the pipeline if no exceptions
            self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)

    def setup_generator(self, model_name):
        token = os.environ.get("HUGGINGFACE_TOKEN")
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     cache_dir="models",
                                                     torch_dtype="auto",
                                                     use_auth_token=token).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        def gen(text, max_new_tokens=100) -> str:
            inputs = tokenizer(text, return_tensors="pt", return_attention_mask=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return text
        return gen

    def generate(self, prompt: str) -> str:
        if hasattr(self, 'generator'):
            return self.generator(prompt, max_new_tokens=100)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=100)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
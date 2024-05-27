from transformers import pipeline, set_seed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod
from requests import Request
import os

class LLM(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, prompt:str) -> str:
        pass

class GeminiLLM(LLM):
    def __init__(self, model_name:str = "gemini-1.5-flash", generation_config:dict = {}):
        try :
            import os
            import google.generativeai as genai
        except:
            raise ImportError("Please install the google-generativeai package to use the Gemini model")

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        generation_config = {
            "temperature": generation_config.get("temperature", 1),
            "top_p": generation_config.get("top_p", 0.95),
            "top_k": generation_config.get("top_k", 50),
            "max_output_tokens": generation_config.get("max_output_tokens", 100),
            "response_mime_type": "text/plain",
        }

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )


    def generate(self, prompt:str) -> str:
        chat_session = self.model.start_chat(
            history=[
            ]
        )
        response = chat_session.send_message(prompt)

        return response.text




def generator(model_name):
    token = os.environ.get("HUGGINGFACE_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                cache_dir="models",
                                                torch_dtype="auto", 
                                                token = token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token)

    def gen(text, max_length=100) -> str:
        inputs = tokenizer(text, return_tensors="pt", return_attention_mask=False)
        outputs = model.generate(**inputs, max_length=max_length)
        text = tokenizer.batch_decode(outputs)[0]
        return text

    return gen


class HuggingFaceLLM(LLM):
    def __init__(self, model:str):
        try:
            self.generator = pipeline('text-generation', model=model)
        except:
            self.generator = generator(model)

    def generate(self, prompt:str) -> str:
        text = self.generator(prompt, max_length=100)[0]
        return text if not 'generated_text' in text else text['generated_text']

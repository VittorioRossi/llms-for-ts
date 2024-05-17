from abc import ABC, abstractmethod
from jinja2 import Template

from prompt.utils import load_template

class LLMForecaster(ABC):
    def __init__(self, prompt_name:str = "base"):
        self.prompt: Template = load_template(prompt_name)
    
    @abstractmethod
    def predict(self, data):
        pass
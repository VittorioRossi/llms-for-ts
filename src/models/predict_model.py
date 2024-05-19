from abc import ABC, abstractmethod
from jinja2 import Template

from prompt.utils import load_template

class LLMForecaster(ABC):
    @abstractmethod
    def predict(self, data):
        pass
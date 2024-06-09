import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from abc import ABC, abstractmethod
import os
import numpy as np
import logging
import regex as re

# Configure the root logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger for the 'models' module
logger = logging.getLogger('models')
logger.setLevel(logging.DEBUG)  # Set the logger to the lowest level to capture all messages

# Remove any existing handlers
for handler in logger.handlers:
    logger.removeHandler(handler)

# Prevent the logger from propagating messages to the root logger
logger.propagate = False

# Create a file handler for logging to a file
file_handler = logging.FileHandler(os.environ.get('LOG_FILE', 'models.log'))
file_handler.setLevel(logging.DEBUG)  # Set the file handler to DEBUG level

# Create a formatter and set it for the file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)
def set_pad_token_if_missing(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def compute_new_tokens(target_size, example_output, tokenizer):
    example_tokens = tokenizer(example_output, add_special_tokens=False)['input_ids']
    return target_size * len(example_tokens)


def extract_numbers(text):
    # Find all sequences of digits with optional decimal points in the text
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    # Convert the sequences of digits to float or int based on their content
    return [float(num) if '.' in num else int(num) for num in numbers]

def clean_pred(pred: str, target_size: int):
    # Extract numbers from the prediction string
    numbers = extract_numbers(pred)
    
    # Initialize the result list
    res = []
    for num in numbers[:target_size]:
        res.append(num)

    # If the number of numbers is less than target_size, append np.nan to the result
    while len(res) < target_size:
        res.append(np.nan)

    logger.info(f'Cleaned prediction: {res}\n')
    return res


class LLM(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, batch: list[str]) -> str:
        pass

class PipelineLLM(LLM):
    def __init__(self, model: str, example_output="00.0", target_size=1, **kwargs):
        self.pipeline = self.setup_pipeline(model, example_output, target_size)
        self.system_message = kwargs.get('system_message', None)

    
    def setup_pipeline(self, model_name, example_output="00.0", target_size=1):
        return pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1
        )

    def get_generation_params(self, max_length, example_output, target_size, batch_size=1, **kwargs):
        max_new_tokens = compute_new_tokens(target_size, example_output, self.pipeline.tokenizer)
        params = {
            "max_length":max_length, 
            'no_repeat_ngram_size':2,
            "return_full_text": False,
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.pipeline.tokenizer.pad_token_id,
            "clean_up_tokenization_spaces": True,
        }
        return params

    def add_system_message(self, message, system_message):
        return [{'role':'user', 'content':system_message}, {'role':'user', 'content':message}]

    def generate(self, message, **kwargs):
        if self.system_message is not None:
            message = self.add_system_message(message, self.system_message)

        params = self.get_generation_params(**kwargs)

        return self.pipeline(message, **params)


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
            try:
                outputs = self.generate_outputs(model, tokenizer, inputs, max_new_tok)
            except Exception as e:
                logger.error(f'Failed to generate outputs for model {model_name}. On input {texts} with error {e}. - Tokenizer input shape {inputs["input_ids"].shape} - Tokenizer attention mask shape {inputs["attention_mask"].shape}')
                return [np.nan for _ in range(target_size)] * len(texts)
            
            results = self.decode_outputs(tokenizer, texts, outputs, target_size=target_size)
            return results

        return gen
    
    def generate(self, batch):
        return self.generator(batch)

    def load_model(self, model_name, token):
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir="models",
            torch_dtype="auto",
            token=token,
        ).to(self.device)

    def load_tokenizer(self, model_name, token):
        return AutoTokenizer.from_pretrained(model_name, 
                                             token=token, 
                                             padding_side='left')

    def tokenize_inputs(self, tokenizer, texts, max_length=4000):
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding='longest',
            truncation='longest_first',
            max_length=max_length,
        )
    
        return {k: v.to(self.device) for k, v in inputs.items()}

    def generate_outputs(self, model, tokenizer, inputs, max_new_tokens):
        return model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    def decode_outputs(self, tokenizer, texts, outputs, target_size):
        generated_texts = tokenizer.batch_decode(outputs, 
                                                 skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=True)
        torch.cuda.empty_cache()

        results = []
        for text, generated_text in zip(texts, generated_texts):

            generated_text = generated_text[len(text):]
            logger.info(f'Generated text: {generated_text}')
            
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
            try:
                outputs = self.generate_outputs(model, tokenizer, inputs_batch, max_new_tok)
            except Exception as e:
                logger.error(f'Failed to generate outputs for model {model_name}. On input {batch_messages} with error {e}. - Tokenizer input shape {inputs_batch["input_ids"].shape} - Tokenizer attention mask shape {inputs_batch["attention_mask"].shape}')
                return [np.nan for _ in range(target_size)] * len(batch_messages)
            
            results = self.decode_outputs(tokenizer, preproces_batch, outputs, target_size=target_size)
            return results

        return gen
    
    
    def apply_system_message(self, batch_messages, system_message = "you are a time series forecasting model"):
        return [
            [
                {'role':'user', 'content':system_message},
                {'role':'user', 'content':message}
            ]
            for message in batch_messages
        ]

    def tokenize_batch(self, tokenizer, batch_messages, max_length=4000):
        def apply_chat_template(messages):
            return tokenizer(
                tokenizer.eos_token.join([msg['content'] for msg in messages]),
                return_tensors="pt",
                padding='longest',
                truncation='longest_first',
                max_length=max_length
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
        generated_texts = tokenizer.batch_decode(outputs, 
                                                 skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=True)
        torch.cuda.empty_cache()

        results = []
        for messages, generated_text in zip(batch_messages, generated_texts):
            original_text = ' '.join([msg['content'] for msg in messages])
            generated_text = generated_text[len(original_text):].strip()

            logger.info(f'Generated text: {generated_text}')

            preds = clean_pred(generated_text, target_size)
            if np.isnan(preds).any():
                print(f"Failed to convert prediction '{generated_text}' to float")

            results.append(preds)

        return results
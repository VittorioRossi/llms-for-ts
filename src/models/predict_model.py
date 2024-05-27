from src.models.models import gpt2
import logging

def main():
    """ Runs model prediction scripts to generate text.
    """
    logger = logging.getLogger(__name__)

    logger.info('Generating text')
    generator = gpt2()
    text = generator("Hello, my name is", max_length=50, num_return_sequences=5)
    logger.info('Text generated')

    return text


if __name__ == '__main__':
    main()
import argparse
import fire
import logging
import sys
from datetime import datetime
import os

from neural_nlp import score as score_function

print(os.environ)

_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
FLAGS, FIRE_FLAGS = parser.parse_known_args()
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(FLAGS.log_level))
_logger.info(f"Running with args {FLAGS}, {FIRE_FLAGS}")
for ignore_logger in ['transformers.data.processors', 'botocore', 'boto3', 'urllib3', 's3transfer']:
    logging.getLogger(ignore_logger).setLevel(logging.INFO)

def run(benchmark, model, layers=None, subsample=None):
    """
    Run full pipeline, but use environment flags to specify
        * how model activations for the sentence are being obtained
        * what coordinate should be used for the splits for the cross validation

    NOTE: Instead of using all these os.environment variables, I could just add arguments to this function which
    would be usable as flags due to fire module; keeping environment variables for now though
    """
    # 1. How to get model activations
    ## 1.a. Sequence summary (default for GPT-2 is last-token emb)
    _logger.info(f"Environment variable AVG_TOKEN_TRANSFORMERS set to: {os.getenv('AVG_TOKEN_TRANSFORMERS')}")

    ## 1.b. Sentence context (default is Passage in this script; default for Schrimpf paper is Topic
    ## (obtained via PAPER_GROUPING))
    _logger.info(f"Environment variable DECONTEXTUALIZED_EMB set to: {os.getenv('DECONTEXTUALIZED_EMB')}")
    _logger.info(f"Environment variable PAPER_GROUPING set to: {os.getenv('PAPER_GROUPING')}")
    if os.getenv('PAPER_GROUPING', '0') == '1' and os.getenv('DECONTEXTUALIZED_EMB', '0') == '1':
        raise ValueError("You cannot have two contradictory context variables defined!")

    # 2. How to do cross_validation
    _logger.info(f"Environment variable SPLIT_AT_PASSAGE set to: {os.getenv('SPLIT_AT_PASSAGE')}")
    _logger.info(f"Environment variable SPLIT_AT_TOPIC set to: {os.getenv('SPLIT_AT_TOPIC')}")
    if os.getenv('SPLIT_AT_PASSAGE', '0') == '1' and os.getenv('SPLIT_AT_TOPIC', '0') == '1':
        raise ValueError("You cannot have two contradictory split_coordinates defined!")

    # Set variables accordingly (used for score saving)
    if os.getenv('DECONTEXTUALIZED_EMB', '0') == '1':
        emb_context = "Sentence"
    elif os.getenv('PAPER_GROUPING', '0') == '1':
        emb_context = "Topic"
    else:
        emb_context = "Passage"

    _logger.info(f"Contextualizing sentence embeddings with context: {emb_context}")

    if os.getenv('SPLIT_AT_PASSAGE', '0') == '1':
        split_coord = "Passage"
    elif os.getenv('SPLIT_AT_TOPIC', '0') == '1':
        split_coord = "Topic"
    else:
        split_coord = "Sentence"

    _logger.info(f"Cross validation split coordinate is: {split_coord}")

    start = datetime.now()
    # Pass arguments on to score_function; these will be used for caching result
    score = score_function(model=model, layers=layers, subsample=subsample, benchmark=benchmark,
                           emb_context=emb_context, split_coord=split_coord)
    end = datetime.now()
    print(score)
    print(f"Duration: {end - start}")


if __name__ == '__main__':
    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)
    fire.Fire(command=FIRE_FLAGS)

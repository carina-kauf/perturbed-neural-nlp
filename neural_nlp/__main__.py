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
    print(f"Environment variable AVG_TOKEN_TRANSFORMERS set to: {os.getenv('AVG_TOKEN_TRANSFORMERS')}")
    print(f"Environment variable SPLIT_AT_PASSAGE set to: {os.getenv('SPLIT_AT_PASSAGE')}")
    print(f"Environment variable SPLIT_AT_TOPIC set to: {os.getenv('SPLIT_AT_TOPIC')}")

    if os.getenv('SPLIT_AT_PASSAGE', '0') == '1' and os.getenv('SPLIT_AT_TOPIC', '0') == '1':
        raise ValueError("You cannot have two contradictory split_coordinates defined!")

    if os.getenv('SPLIT_AT_PASSAGE', '0') == '1' and "passagesplit" not in benchmark:
            raise ValueError("You want the spit_coord to be 'passage_index', but you're not running a PassageSplit benchmark!")
    if os.getenv('SPLIT_AT_TOPIC', '0') == '1' and "topicsplit" not in benchmark:
            raise ValueError("You want the spit_coord to be 'passage_category', but you're not running a TopicSplit benchmark!")
    start = datetime.now()
    score = score_function(model=model, layers=layers, subsample=subsample, benchmark=benchmark)
    end = datetime.now()
    print(score)
    print(f"Duration: {end - start}")


if __name__ == '__main__':
    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)
    fire.Fire(command=FIRE_FLAGS)

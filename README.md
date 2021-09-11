
# Perturbed Language Brain-Score

Code accompanying the paper XXX.

The pipeline is an extension of [neural-nlp](https://github.com/mschrimpf/neural-nlp) (Schrimpf et al., 2021, *XX*).

This pipeline compares X
It builds on the [Brain-Score](www.Brain-Score.org) framework and can easily be extended with new models and datasets.

## Installation
```bash
conda create -n envname python=3.6.2
conda activate envname

git clone https://github.com/carina-kauf/perturbed-neural-nlp.git
cd perturbed-neural-nlp
pip install -e . --use-deprecated=legacy-resolver
```
Using the flag `--use-deprecated=legacy-resolver` immensely speeds up the process and enables a full installation of all required packages/repos. It does result in the following conflict, however, which should be ok for our purposes though:
ERROR: pip's legacy dependency resolver does not consider dependency conflicts when selecting packages. This behaviour is the source of the following dependency conflicts.
pytest 6.2.5 requires py>=1.8.2, but you'll have py 1.8.0 which is incompatible.


Install spacy and nltk via
```bash
pip install spacy==3.1.2
git clone https://github.com/nltk/nltk_contrib.git
```
Then run 
```bash
python -m spacy download en_core_web_sm
```

## Run
To score distilgpt on the Pereira2018-encoding-scrambled1 benchmark (using the *last-token representation* as the sentence representation):

```bash
python neural_nlp run --model distilgpt --benchmark Pereira2018-encoding-scrambled1 --log_level DEBUG
```

To score distilgpt on the Pereira2018-encoding-scrambled1 benchmark (using the *average token representation* as the sentence representation):

```bash
AVG_TOKEN_TRANSFORMERS=1 python neural_nlp run --model distilgpt-avgtoken --benchmark Pereira2018-encoding-scrambled1 --log_level DEBUG
```

Other available benchmarks are e.g. XX

You can also specify different models to run. Currently, all GPT models from [Huggingface Transformers](https://huggingface.co/transformers/) are supported.

## Citation


```

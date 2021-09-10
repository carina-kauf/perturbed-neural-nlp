
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
pip install -e .
```
Install spacy via
```bash
pip install spacy==3.1.2```
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

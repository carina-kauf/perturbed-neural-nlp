
# Perturbed Language Brain-Score

Code accompanying the paper XXX.

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


## Run
To score distilgpt on the Pereira2018-encoding-scrambled1 benchmark:

```bash
python neural_nlp run --model distilgpt --benchmark Pereira2018-encoding-scrambled1 --log_level DEBUG
```

Other available benchmarks are e.g. XX

You can also specify different models to run. Currently, all GPT models from [Huggingface Transformers](https://huggingface.co/transformers/) are supported.

## Citation


```

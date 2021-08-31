
# Language Brain-Score

Code accompanying the paper XXX.

Large-scale evaluation of neural network language models 
as predictive models of human language processing.
This pipeline compares dozens of state-of-the-art models and 4 human datasets (3 neural, 1 behavioral).
It builds on the [Brain-Score](www.Brain-Score.org) framework and can easily be extended with new models and datasets.

## Installation
```bash
conda create -n testenv python=3.6.2
conda activate testenv

git clone https://github.com/carina-kauf/perturbed-neural-nlp.git
cd perturbed-neural-nlp
pip install -e .
```


## Run
To score gpt2-xl on the Blank2014fROI-encoding benchmark:

```bash
python neural_nlp run --model gpt2-xl --benchmark Blank2014fROI-encoding --log_level DEBUG
```

Other available benchmarks are e.g. Pereira2018-encoding (takes a while to compute), and Fedorenko2016v3-encoding.

You can also specify different models to run -- 
note that some of them require additional download of weights (run `ressources/setup.sh` for automated download).

## Citation


```


# Lexical semantic content, not syntactic structure, is the main contributor to ANN-brain similarity of fMRI responses in the language network

Code accompanying the paper [Lexical semantic content, not syntactic structure, is the main contributor to ANN-brain similarity of fMRI responses in the language network](https://www.biorxiv.org/content/10.1101/2023.05.05.539646v1) by Kauf, Tuckute, Levy, Andreas and Fedorenko.

The pipeline is an extension of the [neural-nlp](https://github.com/mschrimpf/neural-nlp) (Schrimpf et al., 2021, *PNAS*). It builds on the [Brain-Score](www.Brain-Score.org) framework and can easily be extended with new models and datasets.

## Abstract
Representations from artificial neural network (ANN) language models have been shown to predict human brain activity in the language network. To understand what aspects of linguistic stimuli contribute to ANN-to-brain similarity, we used an fMRI dataset of responses to n=627 naturalistic English sentences (Pereira et al., 2018) and systematically manipulated the stimuli for which ANN representations were extracted. In particular, we i) perturbed sentences' word order, ii) removed different subsets of words, or iii) replaced sentences with other sentences of varying semantic similarity. We found that the lexical semantic content of the sentence (largely carried by content words) rather than the sentence's syntactic form (conveyed via word order or function words) is primarily responsible for the ANN-to-brain similarity. In follow-up analyses, we found that perturbation manipulations that adversely affect brain predictivity also lead to more divergent representations in the ANN's embedding space and decrease the ANN's ability to predict upcoming tokens in those stimuli. Further, results are robust to whether the mapping model is trained on intact or perturbed stimuli, and whether the ANN sentence representations are conditioned on the same linguistic context that humans saw. The critical result—that lexical-semantic content is the main contributor to the similarity between ANN representations and neural ones—aligns with the idea that the goal of the human language system is to extract meaning from linguistic strings. Finally, this work highlights the strength of systematic experimental manipulations for evaluating how close we are to accurate and generalizable models of the human language network.

## Installation
```bash
conda create -n perturbed-nnlp python=3.8.13
conda activate perturbed-nnlp

git clone https://github.com/carina-kauf/perturbed-neural-nlp.git
cd perturbed-neural-nlp
pip install -r requirements.txt
```

## Run
To score gpt2-xl on the *TrainPerturbed_TestPerturbed_contextualized* Pereira2018-encoding-scrambled1 benchmark (using the last-token representation as the sentence representation) run:

```bash
python neural_nlp run --model gpt2-xl --benchmark Pereira2018-encoding-scrambled1 --log_level DEBUG
```

To score gpt2-xl on the *TrainIntaxt_TestPerturbed_contextualized* Pereira2018-encoding-scrambled1 benchmark (using the last-token representation as the sentence representation) run:

```bash
python neural_nlp run --model gpt2-xl --benchmark Pereira2018-encoding-teston:scr1 --log_level DEBUG
```

All available benchmarks can be found [here](https://github.com/carina-kauf/perturbed-neural-nlp/blob/master/neural_nlp/benchmarks/neural.py#L1589).

Embedding contextualization and cross-validation paradigms are specified via environment variables (see [here](https://github.com/carina-kauf/perturbed-neural-nlp/blob/master/neural_nlp/__main__.py#L37)).

You can also specify different models to run. Currently, all GPT models from [Huggingface Transformers](https://huggingface.co/transformers/) are supported.

## Citation
If you use this work, please cite
```
@article{kauf2023lexical,
  title={Lexical semantic content, not syntactic structure, is the main contributor to ANN-brain similarity of fMRI responses in the language network},
  author={Kauf, Carina and Tuckute, Greta and Levy, Roger and Andreas, Jacob and Fedorenko, Evelina},
  journal={bioRxiv},
  doi = {10.1101/2023.05.05.539646},
  year={2023}
}
```

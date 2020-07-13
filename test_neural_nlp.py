import argparse
from neural_nlp.models.implementations import _PytorchTransformerWrapper, word_last, transformer_configurations, model_layers
from neural_nlp import score  as score_function
import torch
import sys
import os
import numpy as np
import getpass

from transformers import GPT2Tokenizer
from transformers import GPT2Model
from transformers import GPT2Config

user=getpass.getuser()
if user=='eghbalhosseini':
    sys.path.append('/Users/eghbalhosseini/MyCodes/arch_search/arch_search_utils')
    model_and_config_dir = '/Users/eghbalhosseini/MyData/arch_search'
elif user=='ehoseini':
    sys.path.append('/home/ehoseini/MyCodes/arch_search/arch_search_utils')
    model_and_config_dir = '/om/user/ehoseini/MyData/arch_search/'
GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP = {"gpt2": os.path.join(model_and_config_dir,'CONFIG_ARCHIVE_MAP','originals','gpt2-config.json'),
                                      "gpt2-medium": os.path.join(model_and_config_dir,'CONFIG_ARCHIVE_MAP','originals','gpt2-medium-config.json'),
                                      "gpt2-large": os.path.join(model_and_config_dir,'CONFIG_ARCHIVE_MAP','originals','gpt2-large-config.json'),
                                      "gpt2-xl": os.path.join(model_and_config_dir,'CONFIG_ARCHIVE_MAP','originals','gpt2-xl-config.json'),
                                      "distilgpt2": os.path.join(model_and_config_dir,'CONFIG_ARCHIVE_MAP','originals','distilgpt2-config.json')}

GPT2_PRETRAINED_MODEL_ARCHIVE_MAP = {"gpt2": os.path.join(model_and_config_dir,'MODEL_ARCHIVE_MAP','gpt2-pytorch_model.bin'),
                                      "gpt2-medium": os.path.join(model_and_config_dir,'MODEL_ARCHIVE_MAP','gpt2-medium-pytorch_model.bin'),
                                      "gpt2-large": os.path.join(model_and_config_dir,'MODEL_ARCHIVE_MAP','gpt2-large-pytorch_model.bin'),
                                      "gpt2-xl": os.path.join(model_and_config_dir,'MODEL_ARCHIVE_MAP','gpt2-xl-pytorch_model.bin'),
                                      "distilgpt2": os.path.join(model_and_config_dir,'MODEL_ARCHIVE_MAP','distilgpt2-pytorch_model.bin')}

if __name__ =='__main__':
    model='distilgpt2'
    benchmark="Fedorenko2016v3-encoding-weights"
    config_file=GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP[model]
    model_file=GPT2_PRETRAINED_MODEL_ARCHIVE_MAP[model]
    benchmark_tsk = benchmark

    config = GPT2Config.from_json_file(config_file)
    num_layers = config.n_layer
    config.output_hidden_states = True
    config.state_dict = None
    model = GPT2Model(config)
    model.from_pretrained(model_file, config=config)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model_identifier = config.weight_identifier
    config_idx = int(np.argwhere([x['weight_identifier'] == config.weight_identifier for x in transformer_configurations]))
    brainscore_config = transformer_configurations[config_idx]
    brainscore_config['tokenizer_ctr'] = brainscore_config.get('tokenizer_ctr',brainscore_config['prefix'] + 'Tokenizer')
    brainscore_config['model_ctr'] = brainscore_config.get('model_ctr', brainscore_config['prefix'] + 'Model')
    brainscore_config['config_ctr'] = brainscore_config.get('config_ctr', brainscore_config['prefix'] + 'Config')

    transformer = _PytorchTransformerWrapper(identifier=model_identifier,
                                             tokenizer=tokenizer,
                                             tokenizer_special_tokens=brainscore_config.get('tokenizer_special_tokens',
                                                                                            ()),
                                             model=model,
                                             layers=list(brainscore_config['layers']),
                                             sentence_average=word_last)

    score_results=score_function(benchmark=benchmark_tsk, model=model_identifier, model_impl=transformer,
                      layers=list(brainscore_config['layers']))

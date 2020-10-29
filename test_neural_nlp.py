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

if __name__ =='__main__':
    #model_name='distilgpt2'
    #benchmark_tsk="Fedorenko2016v3-encoding-weights"
    benchmark_tsk="Pereira2018-encoding-weights"
    model_name='distilgpt2'

    config = GPT2Config.from_pretrained(model_name)
    num_layers = config.n_layer
    config.output_hidden_states = True
    config.state_dict = None
    model = GPT2Model(config)
    model.from_pretrained(model_name, config=config)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    config.weight_identifier=model_name
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
                      layers=[list(brainscore_config['layers'])[2]])

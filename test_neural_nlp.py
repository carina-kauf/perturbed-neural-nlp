from neural_nlp.models.implementations import _PytorchTransformerWrapper, word_last, transformer_configurations, model_layers
from neural_nlp import score  as score_function
import torch
import sys
import os
import numpy as np


from transformers import GPT2Tokenizer
from transformers import GPT2Model
from transformers import GPT2Config




if __name__ =='__main__':
    model_name='gpt2'
    benchmark_tsk="Pereira2018-encoding-weights"
    #enchmark_tsk = "Fedorenko2016v3-encoding-weights"


    config = GPT2Config.from_pretrained(model_name)
    num_layers = config.n_layer
    config.output_hidden_states = True
    config.state_dict = None
    config.weight_identifier=model_name
    model = GPT2Model(config)
    model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
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

import captum
import torch
from captum.attr import NoiseTunnel
from captum.attr import IntegratedGradients
from captum.attr import LayerGradCam
from captum.attr import GuidedBackprop

from RISE.explanations import RISE


### Wrapper functions ###

# Captum

class SmoothGrad(NoiseTunnel):
    def __init__(self, model, batch_size=16, **kwargs):
        self.saliency = captum.attr.Saliency(model)
        self.batch_size = batch_size
        super().__init__(self.saliency)

    def attribute(self, inputs, target=None):
        return super().attribute(inputs,
                                 target=target,
                                 nt_samples_batch_size=self.batch_size,
                                 **methods_dict['smoothgrad']['params_attr'])


class GradWrapper:
    def __init__(self, model, method_name='integratedgrad', batch_size=16, **kwargs):
        self.method_name = method_name
        self.batch_size = batch_size
        self.xai_method = methods_dict[method_name]['base_class'](model)
        print(self.xai_method)

    def attribute(self, inputs, target=None):
        if methods_dict[self.method_name]['use_batch_size']:
            return  self.xai_method.attribute(inputs,
                                              target=target,
                                              internal_batch_size=self.batch_size,
                                              **methods_dict[self.method_name]['params_attr'])
        else:
            return self.xai_method.attribute(inputs,
                                             target=target,
                                             **methods_dict[self.method_name]['params_attr'])

# Rise

class Rise:
    def __init__(self, model, input_size=224, batch_size=16, **kwargs):
        params = methods_dict['rise']['params_attr']
        self.rise = RISE(model, (input_size, input_size), batch_size)
        self.rise.generate_masks(N=params['n_masks'], s=8, p1=0.1)
        self.input_size = input_size

    def attribute(self, inputs, target=None):
        with torch.no_grad():
            return self.rise(inputs)[target].view((1, 1, self.input_size, self.input_size))


### Parameters for each method ###
methods_dict = {
    'integratedgrad': {
        'class_fn': GradWrapper,
        'base_class': IntegratedGradients,
        'use_batch_size': True,
        'params_attr': {
            'n_steps': 50,
        },
    },
    'smoothgrad': {
        'class_fn': SmoothGrad,
        'params_attr': {
            'nt_samples': 50,
        },
    },
    'guidedbackprop': {
        'class_fn': GradWrapper,
        'base_class': GuidedBackprop,
        'use_batch_size': False,
        'params_attr': {
        },
    },
    'rise': {
        'class_fn': Rise,
        'params_attr': {
            'n_masks': 4000,
        }
    },
}


def get_method(name, model, batch_size=16):
    cur_dict = methods_dict[name]
    return cur_dict["class_fn"](model, method_name=name, batch_size=batch_size)

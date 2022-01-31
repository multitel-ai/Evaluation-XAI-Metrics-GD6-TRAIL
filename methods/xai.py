import captum
import torch
from captum.attr import NoiseTunnel
from captum.attr import IntegratedGradients
from captum.attr import LayerGradCam

from RISE.explanations import RISE

class SmoothGrad(NoiseTunnel):
    def __init__(self, model):
        self.saliency = captum.attr.Saliency(model)
        super().__init__(self.saliency)

    def attribute(self, inputs, target=None):
        return super().attribute(inputs, target=target, **methods_dict['smoothgrad']['params_attr'])

class GradCAM(LayerGradCam):
    def __init__(self, model):
        super().__init__(model, model.layer4)

class Rise:
    def __init__(self, model, input_size=224, batch_size=16):
        params = methods_dict['rise']['params_attr']
        self.rise = RISE(model, (input_size, input_size), batch_size)
        self.rise.generate_masks(N=params['n_masks'], s=8, p1=0.1)
        self.input_size = input_size

    def attribute(self, inputs, target=None):
        with torch.no_grad():
            return self.rise(inputs)[target].view((1, 1, self.input_size, self.input_size))


methods_dict = {
    'integratedgrad': {
        'class_fn': IntegratedGradients,
    },
    'smoothgrad': {
        'class_fn': SmoothGrad,
        'params_attr': {
            'nt_samples': 50,
        },
    },
    'gradcam': {
        'class_fn': GradCAM,
    },
    'rise': {
        'class_fn': Rise,
        'params_attr': {
            'n_masks': 4000,
        }
    },
}


def get_method(name, model):
    cur_dict = methods_dict[name]
    return cur_dict["class_fn"](model)

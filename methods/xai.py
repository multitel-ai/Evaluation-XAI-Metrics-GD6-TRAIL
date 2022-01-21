import captum
from captum.attr import NoiseTunnel
from captum.attr import IntegratedGradients
from captum.attr import LayerGradCam

class SmoothGrad(NoiseTunnel):
    def __init__(self, model):
        self.saliency = captum.attr.Saliency(model)
        super().__init__(self.saliency)

    def attribute(self, inputs, target=None):
        return super().attribute(inputs, target=target, **methods_dict['smoothgrad']['params_attr'])

class GradCAM(LayerGradCam):
    def __init__(self, model):
        super().__init__(model, model.layer4)


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
    }
}


def get_method(name, model):
    cur_dict = methods_dict[name]
    return cur_dict["class_fn"](model)

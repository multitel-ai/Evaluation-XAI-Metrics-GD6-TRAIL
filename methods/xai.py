import captum
import torch
from captum.attr import NoiseTunnel
from captum.attr import IntegratedGradients
from captum.attr import GuidedBackprop

from RISE.explanations import RISE

from torchcam.methods import GradCAM
from torchcam.methods import ScoreCAM
from torchcam.methods import LayerCAM


from torchvision.transforms import functional as Ft

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


# Torchcam

class CAMWrapper:
    def __init__(self,
                 model,
                 method_name='gradcam',
                 batch_size=16,
                 **kwargs):
        self.method_name = method_name
        self.model = model
        if methods_dict[method_name]['use_batch_size']:
            self.xai_method = methods_dict[method_name]['base_class'](model, batch_size=batch_size)
        elif method_name == 'layercam':
            self.xai_method = methods_dict[method_name]['base_class'](model,
                                                                      target_layer=methods_dict[method_name]['layers'])
        else:
            self.xai_method = methods_dict[method_name]['base_class'](model)

    def attribute(self, inputs, target=None):
        torch.set_grad_enabled(True)
        input_grad = inputs.clone().detach().requires_grad_(True)
        out = self.model(input_grad)
        map = self.xai_method(target.item(), out)
        if self.method_name == 'layercam':
            map = self.xai_method.fuse_cams(map)
            map = map.view(1, 1, *map.shape)
        else:
            map = map[0].view(1, 1, *map[0].shape)
        return map


### Dummy xai methods for sanity check ###

class Random:
    """
    Output a random map with the same size as the input but 1 channel
    """
    def __init__(self, model, **kwargs):
        pass

    def attribute(self, inputs, target=None):
        return torch.rand(1, 1, *inputs.shape[-2:])

class Sobel:
    """
    Border detection using a Sobel filter
    """
    def __init__(self, model, **kwargs):
        pass

    def attribute(self, inputs, target=None):
        grayscale = Ft.rgb_to_grayscale(inputs)
        sobel_filter = torch.tensor(
            [[[[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]], ],
             [[[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]]], ],
            device=inputs.device,
            dtype=inputs.dtype
        )
        x = torch.nn.functional.conv2d(grayscale, sobel_filter)
        x = x.square()
        x = x.sum(1, keepdim=True)
        x = x.sqrt()
        return x


class Gaussian:
    def __init__(self, model, **kwargs):
        pass

    def attribute(self, inputs, target=None):
        # Get input sizes
        x_size, y_size = inputs.shape[-2:]

        # create a 2d meshgrid from -1 to 1 with same size as input
        x, y = torch.meshgrid(torch.linspace(-1, 1, x_size, device=inputs.device),
                              torch.linspace(-1, 1, y_size, device=inputs.device))

        # distance from the mean (center)
        distance = torch.sqrt(x * x + y * y)

        # Set sigma to 1 (mu = 0)
        sigma = 1

        # Calculating Gaussian array
        gaussian_2d = torch.exp(- (distance ** 2 / (2.0 * sigma ** 2)))

        return gaussian_2d.view(1, 1, x_size, y_size)


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
    'gradcam': {
        'class_fn': CAMWrapper,
        'base_class': GradCAM,
        'use_batch_size': False,
    },
    'scorecam': {
        'class_fn':CAMWrapper,
        'base_class': ScoreCAM,
        'use_batch_size': True,
    },
    'layercam': {
        'class_fn':CAMWrapper,
        'base_class': LayerCAM,
        'use_batch_size': False,
        'layers': ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
    },
    'random': {
        'class_fn': Random,
    },
    'sobel': {
        'class_fn': Sobel,
    },
    'gaussian': {
        'class_fn': Gaussian,
    }
}


def get_method(name, model, batch_size=16):
    cur_dict = methods_dict[name]
    return cur_dict["class_fn"](model, method_name=name, batch_size=batch_size)

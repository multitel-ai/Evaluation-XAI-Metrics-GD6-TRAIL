import captum
import torch
from captum.attr import NoiseTunnel
from captum.attr import IntegratedGradients
from captum.attr import GuidedBackprop

# Try to load RISE, return warning if not available
try:
    from RISE.explanations import RISE
except:
    print("RISE not installed. You will not be able to use it")

from torchcam.methods import GradCAM
from torchcam.methods import ScoreCAM
from torchcam.methods import LayerCAM

import torchvision
import PyTorch_CIFAR10

import sys

# Try to load polycam, return warning if not available
try:
    sys.path.append("polycam")
    from polycam.polycam import  PCAMpm
except:
    print("Polycam not installed. You will not be able to use it")

# Try to load CAMERAS, return warning if not available
try:
    from CAMERAS.CAMERAS import CAMERAS
except:
    print("CAMERAS not installed. You will not be able to use it")

try:
    from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward
except:
    print("TorchRay not installed. You will not be able to use extremal perturbation")



from torchvision.transforms import functional as Ft

### Wrapper functions ###
# wrap the saliency method into a common shape with attribute method (similar to Captum)

# Captum

class SmoothGrad(NoiseTunnel):
    """
    SmoothGrad method using noise tunnel and saliency method
    Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017). Smoothgrad: removing noise by adding noise.
    arXiv preprint arXiv:1706.03825. https://arxiv.org/abs/1706.03825
    """
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
    """
    Wrapper for gradient based methods (IntegratedGrad and GuidedBackProp, ...)
    """
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
    """
    Wrapper for RISE method
    Petsiuk, V., Das, A., & Saenko, K. (2018). Rise: Randomized input sampling for explanation of black-box models.
    arXiv preprint arXiv:1806.07421. https://arxiv.org/abs/1806.07421
    """
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
    """
    Wrapper for the CAM based methods from TorchCAM ( https://github.com/frgfm/torch-cam )s
    """
    def __init__(self,
                 model,
                 method_name='gradcam',
                 batch_size=16,
                 **kwargs):
        self.method_name = method_name
        self.model = model
        if methods_dict[method_name]['use_batch_size']:
            self.xai_method = methods_dict[method_name]['base_class'](model,
                                                                      batch_size=batch_size,
                                                                      target_layer = models_dict[type(model)]['layers'][-1])
        elif method_name == 'layercam':
            self.xai_method = methods_dict[method_name]['base_class'](model,
                                                                      target_layer=models_dict[type(model)]['layers'])
        else:
            self.xai_method = methods_dict[method_name]['base_class'](model,
                                                                      target_layer=models_dict[type(model)]['layers'][-1])

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
        map = map.nan_to_num()
        return map


### PolyCAM

class PolyCAMWrapper:
    """
    Wrapper for PolyCAM
    https://github.com/andralex8/polycam
    """
    def __init__(self,
                 model,
                 batch_size=16,
                 **kwargs):
        self.pcam = PCAMpm(model, batch_size=batch_size, target_layer_list=models_dict[type(model)]['layers'])

    def attribute(self, inputs, target=None):
        map = self.pcam(inputs, class_idx=target)[-1]
        return map.detach()


### CAMERAS

class CAMERASWrapper:
    """
    Wrapper for CAMERAS
    https://github.com/VisMIL/CAMERAS
    """
    def __init__(self,
                 model,
                 **kwargs):
        self.cameras = CAMERAS(model=model, targetLayerName=models_dict[type(model)]['layers'][-1])

    def attribute(self, inputs, target=None):
        map = self.cameras.run(inputs, classOfInterest=target)
        map = map.view(1, 1, *map.shape)
        return map.detach()


### Extremal perturbation

class EPWrapper:
    """
    Wrapper for Extremal perturbation from TorchRay
    Use the technique described by the authors to obtain a saliency maps by fusing multiple masks
    https://arxiv.org/pdf/1910.08485.pdf
    """
    def __init__(self,
                 model,
                 **kwargs):
        self.model = model

    def attribute(self, inputs, target=None):
        masks, _ = extremal_perturbation(
            self.model, inputs, int(target),
            reward_func=contrastive_reward,
            areas=methods_dict["extremal_perturbation"]["areas"],
        )
        # Sum the masks for multiple areas to get a saliency map
        saliency = masks.sum(0)

        # Gaussian filter with standard deviation equal to 9% of the shorter size of the image
        ## Get a the size of 9% the min size of the image
        kernel_size = int(min(inputs.shape[-2:])*0.09)
        ## Ensure kernel is odd number and convert to tuple
        kernel_size = kernel_size + (1 - kernel_size % 2)
        kernal_size = [kernel_size, kernel_size]

        ## Smooth with gaussian filter
        saliency = torchvision.transforms.functional.gaussian_blur(saliency, kernel_size=kernal_size)

        return saliency.view(1, 1, *saliency.shape[-2:])


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
    """
    Centered Gaussian 2D
    """
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
    },
    'polycam': {
        'class_fn': PolyCAMWrapper,
    },
    'cameras': {
        'class_fn': CAMERASWrapper,
    },
    'extremal_perturbation': {
        'class_fn': EPWrapper,
        'areas': [0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
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

models_dict = {
    torchvision.models.resnet.ResNet: {
        'layers': ['relu', 'layer1', 'layer2', 'layer3', 'layer4'],
    },
    torchvision.models.vgg.VGG: {
        'layers': ['features.3', 'features.8', 'features.15', 'features.22', 'features.29'],
    },
    PyTorch_CIFAR10.cifar10_models.vgg.VGG: {
        'layers': ['features.5', 'features.12', 'features.22', 'features.32', 'features.42'],
    },
    PyTorch_CIFAR10.cifar10_models.resnet.ResNet: {
        'layers': ['relu', 'layer1', 'layer2', 'layer3', 'layer4'],
    }
    
}


def get_method(name, model, batch_size=16):
    """
    Get the corresponding method
    :param name: name of the method to return
    :param model: model to explain
    :param batch_size: size of the internal batch (used by methods using internal batch size)
    :return: method class
    """
    cur_dict = methods_dict[name]
    return cur_dict["class_fn"](model, method_name=name, batch_size=batch_size)

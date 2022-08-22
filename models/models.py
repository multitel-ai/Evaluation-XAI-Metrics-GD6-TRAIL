import torch

from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16

models_dict = {
    'resnet50': {
        'class_fn': resnet50,
        'url': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    },
    'vgg16': {
        'class_fn': vgg16,
        'url': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    },
}


def get_model(name, n_output, checkpoint=None, pretrained=True):
    cur_dict = models_dict[name]

    model = cur_dict['class_fn'](pretrained=False)

    if not checkpoint and pretrained:
        state_dict = torch.hub.load_state_dict_from_url(cur_dict['url'])
        model.load_state_dict(state_dict)

    # change classifier to the correct size
    print("Creating a new FC layer...")
    if name == "resnet50":
        model.fc = torch.nn.Linear(model.fc.in_features, n_output)
    elif name == "vgg16":
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, n_output)

    # load checkpoint
    if checkpoint:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)

    return model
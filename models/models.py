import torch

from torchvision.models.resnet import resnet50

models_dict = {
    'resnet50': {
        'class_fn': resnet50,
        'url': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    },
}

def get_model(name, n_output, checkpoint=None, pretrained=True):
    cur_dict = models_dict[name]

    # use pretrained network only in absence of checkpoint
    if checkpoint:
        model = cur_dict['class_fn'](pretrained=False)
    else:
        print("Loading the pretrained model...")
        model = cur_dict['class_fn'](pretrained=pretrained)

    # change classifier to the correct size
    if model.fc.out_features != n_output:
        print("Creating a new FC layer...")
        model.fc = torch.nn.Linear(model.fc.in_features, n_output)

    # load checkpoint
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))

    return model
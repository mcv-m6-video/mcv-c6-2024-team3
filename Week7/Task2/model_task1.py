'''
model_name = 'x3d_m'
model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)

from torch import nn
model.blocks[5].proj = nn.Identity()
model.blocks[5].output_pool = nn.Identity()
device = 'cuda'
model.to(device)
input_data = torch.randn(1, 3, 20, 540, 540).to(device)
output = model(input_data)
torch.Size([1, 2048])
'''
import torch
from torch import nn

import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LateFusion(nn.Module):
    def __init__(self, num_classes, model_image, model_modality):
        super(LateFusion, self).__init__()

        self.model_image = model_image
        self.model_modality = model_modality
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(2*self.num_classes, 2*self.num_classes),
            nn.ReLU(),
            nn.Linear(2*self.num_classes, 2*self.num_classes),
            nn.ReLU(),
            nn.Linear(2*self.num_classes, 2*self.num_classes),
            nn.ReLU(),
            # ADDED AFTER THE INAKI TRAINING
            nn.Linear(2*self.num_classes, 2*self.num_classes),
            nn.ReLU(),
            nn.Linear(2*self.num_classes, 2*self.num_classes),
            nn.ReLU(),
            # END
            nn.Linear(2*self.num_classes, self.num_classes)
        )

    def forward(self, x_img, x_mod):
        x_img = self.model_image(x_img)
        x_mod = self.model_modality(x_mod)
        x = torch.cat((x_img, x_mod), dim=1)
        x = self.mlp(x)
        return x



def model_creator(num_classes,weights_path):

    # IMAGE MODEL
    model_name_img = 'x3d_m'
    model_image = torch.hub.load('facebookresearch/pytorchvideo', model_name_img, pretrained=True)
    model_image.blocks[5].proj = nn.Identity()
    model_image =  nn.Sequential(
        model_image,
        nn.Linear(2048, num_classes, bias=True),
    )
    model_image.to(device)
    model_image.load_state_dict(torch.load(weights_path[0], map_location=torch.device(device)))
    model_image.eval()

    # MODALITY MODEL
    model_name_modality = 'x3d_xs'
    model_mod = torch.hub.load('facebookresearch/pytorchvideo', model_name_modality, pretrained=True)
    model_mod.blocks[5].proj = nn.Identity()
    model_mod =  nn.Sequential(
        model_mod,
        nn.Linear(2048, num_classes, bias=True),
    )
    model_mod.to(device)
    model_mod.load_state_dict(torch.load(weights_path[1], map_location=torch.device(device)))
    model_mod.eval()
 
    model = LateFusion(num_classes, model_image, model_mod)
    model.to(device)

    '''
    for param in model.model_image.parameters():
        param.requires_grad = False
    
    for param in model.model_modality.parameters():
        param.requires_grad = False
    ''' 

    unfreezed = 50
    total = len(list(model.model_image.parameters()))
    for i, param in enumerate(model.model_image.parameters()):
        if i < (total - unfreezed):
            param.requires_grad = False
        else:
            param.requires_grad = True

    total = len(list(model.model_modality.parameters()))
    for i, param in enumerate(model.model_modality.parameters()):
        if i < (total - unfreezed):
            param.requires_grad = False
        else:
            param.requires_grad = True

    return model

if __name__ == '__main__':
    num_classes = 51
    weigths_path = [
            '/ghome/group03/C6/week7PS/model_weights/image_model.pth',
            '/ghome/group03/C6/week7PS/model_weights/modality_model.pth'
    ]    
    model = model_creator(num_classes,weigths_path)

    print(model)
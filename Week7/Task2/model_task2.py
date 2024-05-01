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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EarlyFusion(nn.Module):
    def __init__(self, num_classes, model_image, model_modality, hidden_size, num_heads, dropout):
        super(EarlyFusion, self).__init__()

        self.model_image = model_image
        self.model_modality = model_modality
        self.num_classes = num_classes
        # Transformer Encoder
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size*4,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=6
        )

        # Output Layer
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x_img, x_mod):
        x_img = self.model_image(x_img)
        x_mod = self.model_modality(x_mod)
    
        combined_features = torch.cat((x_img.unsqueeze(1), x_mod.unsqueeze(1)), dim=1)

        combined_features = combined_features.reshape(x_img.shape[0], -1, 512)
        transformer_output = self.transformer_encoder(combined_features)
        pooled_output = torch.mean(transformer_output, dim=1)
        output_logits = self.output_layer(pooled_output)

        return output_logits



def model_creator(num_classes,weights_path,hidden_size, num_heads, dropout):

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

    model_image = model_image[0]


    model_image.blocks[5].proj = nn.Identity()

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

    model_mod = model_mod[0]


    model_mod.blocks[5].proj = nn.Identity()
 
    model = EarlyFusion(num_classes, model_image, model_mod,hidden_size, num_heads, dropout)
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
    model = model_creator(num_classes,weigths_path, 512, 1, 0.1)

    input_model_img = torch.randn(15, 3, 20, 540, 540).to(device)
    input_model_mod = torch.randn(15, 3, 20, 540, 540).to(device)

    output = model(input_model_img, input_model_mod)
    print(output.shape)
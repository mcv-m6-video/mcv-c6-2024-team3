""" Functions to create models """

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
import torch.nn.utils.prune as prune
from movinets import MoViNet
from movinets.config import _C

def create(model_name: str, load_pretrain: bool, num_classes: int) -> nn.Module:
    if model_name == 'x3d_xs':
        return create_x3d_xs(load_pretrain, num_classes)
    elif model_name == 'movinet':
        return create_movinet(load_pretrain, num_classes)
    
    else:
        raise ValueError(f"Model {model_name} not supported")

def modify_x3d_model(model, reduction_factor=2):
    # Reduce the channels in each layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d):
            module.out_channels = int(module.out_channels / reduction_factor)
            module.in_channels = int(module.in_channels / reduction_factor)
        elif isinstance(module, torch.nn.BatchNorm3d):
            module.num_features = int(module.num_features / reduction_factor)
        elif isinstance(module, torch.nn.Linear):
            module.in_features = int(module.in_features / reduction_factor)
    return model

def create_x3d_xs(load_pretrain, num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=load_pretrain)
    model.blocks[5].pool.post_conv = nn.Identity()
    model.blocks[5].pool.post_conv = nn.Conv3d(432, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
    # print(model)
    # Apply pruning to each module of the model
    # for module_name, module in model.named_modules():
    #     if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
    #     #     print(module)
    #         prune.l1_unstructured(module, name="weight", amount=0.99)  # Example: 20% pruning for demonstration
    #         # Run the pruning
    #         prune.remove(module, 'weight')

    # parameters_to_prune = [
    # (module, "weight") for module in filter(lambda m: type(m) == torch.nn.Conv2d, model.modules())
    # ]
    # print(parameters_to_prune)
    # prune.global_unstructured(
    #     parameters_to_prune,
    #     pruning_method=prune.L1Unstructured,
    #     amount=0.2,
    # )

    


    # Apply pruning to the new classifier
    # prune.l1_unstructured(new_classifier, name="weight", amount=0.99)  # Example: 20% pruning for demonstration
    # prune.remove(new_classifier, 'weight')
    # Apply pruning to the new classifier

    # model = modify_x3d_model(model, reduction_factor=2)
    # print(model)
    model.blocks[5].proj = nn.Identity()

    model.blocks[5].proj = nn.Linear(256, num_classes, bias=True)
    # prune.l1_unstructured(new_classifier, name="weight", amount=0.2)  # Example: 20% pruning for demonstration

    # # Quantize the model
    # model = quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
    
    # # Define example data
    # example_input = torch.rand(1, 3, 4, 182, 182)

    # # Fuse modules and prepare the model for quantization
    # model.fuse_model()

    # model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    # torch.quantization.prepare(model, inplace=True)

    # # Calibration
    # with torch.no_grad():
    #     model(example_input)

    # # Convert the model to a quantized model
    # torch.quantization.convert(model, inplace=True)

    return model

def create_movinet(load_pretrain, num_classes):
    model = MoViNet(_C.MODEL.MoViNetA0, causal = True, pretrained = load_pretrain )

    # print(model.blocks[-1])
    for i in range(1,5):
        # print(i)
        model.blocks[-i] = torch.nn.Identity()
    # print(model.blocks[-1])
    # for name, module in model.blocks[3].named_modules():
    #     print(module)
    #     module = torch.nn.Identity()
    #     print(module)
    # print(model)
    # model = modify_x3d_model(model)
    model.conv7.conv_1.conv2d = torch.nn.Conv2d(56, 480, kernel_size=(1, 1), stride=(1, 1), bias = False)
    model.classifier[0].conv_1.conv2d = torch.nn.Conv2d(480, 64, kernel_size=(1, 1), stride=(1, 1))
    model.classifier[3] = torch.nn.Conv3d(64, num_classes, (1,1,1))

    model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)

    return model
    
    
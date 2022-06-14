import torch
import torch.nn as nn 
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def get_activation_function(name: str = "ReLU"):
    func = {
        "ReLU": nn.ReLU,
        "ELU": nn.ELU,
        "CELU": nn.CELU
    }
    return func[name]()

def sequential_conv_1d_embedding(
    encoder_patch_color: int,
    embedding_size: int,
    encoder_conv_kernel_size: int,
    hidden_sizes: List[int],
    encoder_conv_padding: List[int],
    encoder_conv_stride: List[int],
    batch_norm: bool = False,
    shrink: bool = False,
    activation_function: nn.Module = nn.ReLU()
    
) -> nn.Sequential:
    #print(encoder_conv_padding)
    channel_sizes = hidden_sizes + [embedding_size]

    conv_layers = [nn.Conv2d(encoder_patch_color,
        channel_sizes[0],
        encoder_conv_kernel_size,
        encoder_conv_stride[0],
        encoder_conv_padding[0])]
    
    conv_layers.append(activation_function)
    if (batch_norm):
            conv_layers.append(nn.BatchNorm2d(channel_sizes[0]))

    for i in range(len(channel_sizes)-1):
        conv_layers.append(
            nn.Conv2d(
                channel_sizes[i],
                channel_sizes[i+1],
                encoder_conv_kernel_size,
                encoder_conv_stride[i+1],
                encoder_conv_padding[i+1]
            )
        )
        conv_layers.append(activation_function)
        if (batch_norm):
            conv_layers.append(nn.BatchNorm2d(channel_sizes[i+1]))
    
    if shrink==True:
        conv_layers.append(nn.AdaptiveMaxPool2d((1,1)))
            
    #print(conv_layers)
    return nn.Sequential(*conv_layers)


def sequential_conv_2d_embedding(
    encoder_patch_color: int,
    embedding_size: int,
    encoder_conv_kernel_size: int,
    hidden_sizes: List[int],
    encoder_conv_padding: List[int],
    encoder_conv_stride: List[int],
    batch_norm: bool = False,
    activation_function: nn.Module = nn.ReLU()
    
) -> nn.Sequential:
    print(encoder_conv_padding)
    channel_sizes = hidden_sizes + [1]

    conv_layers = [nn.Conv2d(encoder_patch_color,
        channel_sizes[0],
        encoder_conv_kernel_size,
        encoder_conv_stride[0],
        encoder_conv_padding[0])]
    
    conv_layers.append(activation_function)
    if (batch_norm):
            conv_layers.append(nn.BatchNorm2d(channel_sizes[0]))

    for i in range(len(channel_sizes)-1):
        conv_layers.append(
            nn.Conv2d(
                channel_sizes[i],
                channel_sizes[i+1],
                encoder_conv_kernel_size,
                encoder_conv_stride[i+1],
                encoder_conv_padding[i+1]
            )
        )
        conv_layers.append(activation_function)
        if (batch_norm):
            conv_layers.append(nn.BatchNorm2d(channel_sizes[i+1]))
    
    
    conv_layers.append(nn.AdaptiveMaxPool2d((embedding_size, embedding_size)))
            
    print(conv_layers)
    return nn.Sequential(*conv_layers)

    

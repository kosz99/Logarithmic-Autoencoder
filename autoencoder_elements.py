import torch
import torch.nn as nn 
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from utils import (get_activation_function,
                  sequential_conv_1d_embedding,
                  sequential_conv_2d_embedding)




class LogEncoderEntryLayerConv_1d_embedding(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        encoder_patch_size: int,
        encoder_patch_color: int,
        encoder_conv_kernel_size: int,
        encoder_conv_channels_size: List[int],
        encoder_conv_padding: List[int],
        encoder_conv_stride: List[int],
        encoder_activation_function: str = "ReLU",
        batch_norm: bool = False
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.encoder_patch_size = encoder_patch_size
        self.encoder_patch_color = encoder_patch_color
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_channels_size = encoder_conv_channels_size
        self.encoder_conv_padding = encoder_conv_padding
        self.encoder_conv_stride = encoder_conv_stride
        self.encoder_conv_activation_function = encoder_activation_function
        self.batch_norm = batch_norm


        self.activation_f = get_activation_function(self.encoder_conv_activation_function)

        self.conv_nn = sequential_conv_1d_embedding(
            self.encoder_patch_color,
            self.embedding_size,
            self.encoder_conv_kernel_size,
            self.encoder_conv_channels_size,
            self.encoder_conv_padding,
            self.encoder_conv_stride,
            self.batch_norm,
            self.activation_f
        )

    
    def forward(self, input_tensor):
        # input tensor shape: N x C x H x W
        #print(input_tensor.shape)
        # input tensor shape: N x C x H' x W' x kH x kW
        input_tensor = input_tensor.unfold(-2, self.encoder_patch_size, self.encoder_patch_size)
        input_tensor = input_tensor.unfold(-2, self.encoder_patch_size, self.encoder_patch_size)
        #print(input_tensor.shape)
        # input tensor shape: N x H' x W' x C x kH x kW
        input_tensor = input_tensor.movedim(1, -3)
        #print(input_tensor.shape)
        # input tensor shape: N * H' * W' x C x kH x kW
        shape = input_tensor.size()
        input_tensor = input_tensor.flatten(0,2)
        #print(input_tensor.shape)
        # conv embeddings shape: N * H' * W' x embedding_size x kH x kW
        conv_embeddings = self.conv_nn(input_tensor)
        #print(conv_embeddings.shape)
        # conv embeddings shape: N x H' x W' x embedding_size x 1 x 1
        conv_embeddings = conv_embeddings.view(shape[0], shape[1], shape[2], self.embedding_size)
        #print(conv_embeddings.shape)
        # output embeddings shape: N x H' x W' x embedding_size
        #output_embeddings = torch.squeeze(conv_embeddings, 0)

        return conv_embeddings




class LogEncoderRecursiveLayerConv_1d_embedding(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        encoder_patch_size: int,
        encoder_conv_kernel_size: int,
        encoder_conv_channels_size: List[int],
        encoder_conv_padding: List[int],
        encoder_conv_stride: List[int],
        encoder_activation_function: str = "ReLU",
        batch_norm: bool = False
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.encoder_patch_size = encoder_patch_size
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_channels_size = encoder_conv_channels_size
        self.encoder_conv_padding = encoder_conv_padding
        self.encoder_conv_stride = encoder_conv_stride
        self.encoder_conv_activation_function = encoder_activation_function
        self.batch_norm = batch_norm



        self.activation_f = get_activation_function(self.encoder_conv_activation_function)

        self.conv_nn = sequential_conv_1d_embedding(
            self.embedding_size,
            self.embedding_size,
            self.encoder_conv_kernel_size,
            self.encoder_conv_channels_size,
            self.encoder_conv_padding,
            self.encoder_conv_stride,
            self.batch_norm,
            self.activation_f
        )

    
    def forward(self, input_tensor):
        # input tensor shape: N x H x W x embedding
        #print(input_tensor.shape)
        # input tensor shape: N x H' x W' x embedding x kH x kW
        input_tensor = input_tensor.unfold(-3, self.encoder_patch_size, self.encoder_patch_size)
        input_tensor = input_tensor.unfold(-3, self.encoder_patch_size, self.encoder_patch_size)
        #print(input_tensor.shape)
        # input tensor shape: N * H' * W' x embedding x kH x kW
        shape = input_tensor.size()
        input_tensor = input_tensor.flatten(0,2)
        #print(input_tensor.shape)
        # conv embeddings shape: N * H' * W' x embedding_size x ? x ?
        conv_embeddings = self.conv_nn(input_tensor)
        #print(conv_embeddings.shape)
        # conv embeddings shape: N x H' x W' x embedding_size x 1 x 1
        conv_embeddings = conv_embeddings.view(shape[0], shape[1], shape[2], self.embedding_size)
        #print(conv_embeddings.shape)
        # output embeddings shape: N x H' x W' x embedding_size
        #output_embeddings = torch.squeeze(conv_embeddings, 0)

        return conv_embeddings





class LogEncoderEntryLayerConvModel_1d_embedding(nn.Module):
    def __init__(
            self,
            embedding_size: int,
            encoder_patch_size: int,
            model
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.encoder_patch_size = encoder_patch_size
        self.conv_nn = model
        self.adaptive = nn.AdaptiveAvgPool2d((1,1))

    
    def forward(self, input_tensor):
        # input tensor shape: N x C x H x W
        #print(input_tensor.shape)
        # input tensor shape: N x C x H' x W' x kH x kW
        input_tensor = input_tensor.unfold(-2, self.encoder_patch_size, self.encoder_patch_size)
        input_tensor = input_tensor.unfold(-2, self.encoder_patch_size, self.encoder_patch_size)
        #print(input_tensor.shape)
        # input tensor shape: N x H' x W' x C x kH x kW
        input_tensor = input_tensor.movedim(1, -3)
        #print(input_tensor.shape)
        # input tensor shape: N * H' * W' x C x kH x kW
        shape = input_tensor.size()
        input_tensor = input_tensor.flatten(0,2)
        #print(input_tensor.shape)
        # conv embeddings shape: N * H' * W' x embedding_size x kH x kW
        conv_embeddings = self.conv_nn(input_tensor)
        #conv_embeddings = self.adaptive(conv_embeddings)
        #print(conv_embeddings.shape)
        #print(shape)
        # conv embeddings shape: N x H' x W' x embedding_size x 1 x 1
        conv_embeddings = conv_embeddings.view(shape[0], shape[1], shape[2], self.embedding_size)
        #print(conv_embeddings.shape)
        # output embeddings shape: N x H' x W' x embedding_size
        #output_embeddings = torch.squeeze(conv_embeddings, 0)

        return conv_embeddings


class LogEncoderRecursiveLayerConvModel_1d_embedding(nn.Module):


    def __init__(
        self,
        embedding_size: int,
        encoder_patch_size: int,
        model: nn.Module
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.encoder_patch_size = encoder_patch_size
        self.conv_nn = model
        self.adaptive = nn.AdaptiveAvgPool2d((1,1))

    
    def forward(self, input_tensor):
        # input tensor shape: N x H x W x embedding
        #print(input_tensor.shape)
        # input tensor shape: N x H' x W' x embedding x kH x kW
        input_tensor = input_tensor.unfold(-3, self.encoder_patch_size, self.encoder_patch_size)
        input_tensor = input_tensor.unfold(-3, self.encoder_patch_size, self.encoder_patch_size)
        #print(input_tensor.shape)
        # input tensor shape: N * H' * W' x embedding x kH x kW
        shape = input_tensor.size()
        input_tensor = input_tensor.flatten(0,2)
        #print(input_tensor.shape)
        # conv embeddings shape: N * H' * W' x embedding_size x ? x ?
        conv_embeddings = self.conv_nn(input_tensor)
        #conv_embeddings = self.adaptive(conv_embeddings)
        #print(conv_embeddings.shape)
        # conv embeddings shape: N x H' x W' x embedding_size x 1 x 1
        conv_embeddings = conv_embeddings.view(shape[0], shape[1], shape[2], self.embedding_size)
        #print(conv_embeddings.shape)
        # output embeddings shape: N x H' x W' x embedding_size
        #output_embeddings = torch.squeeze(conv_embeddings, 0)

        return conv_embeddings





class LogDecoderRecursiveLayerConvModel_1_embedding(nn.Module):
    def __init__(self, patch_size, embedding_size, model, last = False):
        super().__init__()
        self.embedding_size = embedding_size
        self.patch_size = patch_size
        self.deconv = nn.ConvTranspose2d(self.embedding_size, self.embedding_size, self.patch_size, self.patch_size)
        self.model = model
        self.last = last

    def forward(self, x):
        x = x.unfold(-2, 1, 1).unfold(-2, 1, 1)
        x = x.movedim(1, -3)
        shape = x.size()
        x = x.flatten(0,2)
        x = self.deconv(x)
        x = self.model(x)
        if self.last == True:
            x = x.view(shape[0], shape[1], shape[2], 3, self.patch_size, self.patch_size)
        else:
            x = x.view(shape[0], shape[1], shape[2], self.embedding_size, self.patch_size, self.patch_size)
        x = x.flatten(1,2)
        x = x.flatten(2,4)
        x = x.permute(0,2,1)
        output = nn.Fold(output_size=(shape[1]*self.patch_size, shape[2]*self.patch_size), kernel_size = (self.patch_size, self.patch_size), stride = (self.patch_size, self.patch_size))(x)
            
        return output






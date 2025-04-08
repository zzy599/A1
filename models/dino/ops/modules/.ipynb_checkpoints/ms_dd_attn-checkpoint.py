# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

import matplotlib.pyplot as plt




def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDDAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=1, kernel_size=3, dilation=[1, 2, 3],
                 attn_drop=0., qk_scale=None):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")


        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.dilation = dilation  # 膨胀率
        self.kernel_size = kernel_size  # 核大小3
        self.scale = qk_scale or _d_per_head ** -0.5
        self.num_dilation = len(dilation)
        self.dilate_attention = nn.ModuleList(                 # 循环访问列表中指定的膨胀率，并为每个膨胀率创建一个模块
            [DDAttention(_d_per_head, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.sampling_offsets = nn.Linear(d_model, n_heads * self.num_dilation * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * self.num_dilation * n_levels * n_points * kernel_size *kernel_size)
        self.value_proj = nn.Linear(d_model, d_model)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.se = nn.Linear(32, self.num_dilation)  # 每个头的通道数为 32
        self.out = nn.Linear(d_model*3, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads * self.num_dilation, dtype=torch.float32) * (2.0 * math.pi / (self.n_heads* self.num_dilation))
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads * self.num_dilation, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        constant_(self.se.weight.data, 0.)
        constant_(self.se.bias.data, 0.)
        xavier_uniform_(self.out.weight.data)
        constant_(self.out.bias.data, 0.)
        


    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes,level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
    
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.num_dilation, self.n_heads, self.n_levels, self.n_points, 2)  
        attention_weights = self.attention_weights(query).view(N, Len_q, self.num_dilation,self.n_heads, self.n_levels * self.n_points * self.kernel_size * self.kernel_size)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.num_dilation,self.n_heads, self.n_levels, self.n_points* self.kernel_size * self.kernel_size)   

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :,None, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None,None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None,None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None,None, :, None, 2:] * 0.5    #b,hw,8,4,1,2
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        v = value.permute(0, 2, 3, 1)   
        x = v.reshape(N*8, self.d_model // self.n_heads, -1)          
        x = self.gap(x).squeeze(-1)  
        scores = self.se(x) 
        scores = scores.view(N, self.n_heads,-1)  
        probs = F.softmax(scores, dim=-1) 
        groups = []
        for g in range(self.num_dilation):
            mask = probs[:, :, g].reshape(N, self.n_heads, 1, 1)   # 获取第 g 组的概率 
            group = v * mask       
            groups.append(group)
    
        sampling_locations = sampling_locations.reshape([N, Len_q, self.num_dilation, self.n_heads, self.n_levels, self.n_points * 2]).permute(2, 0, 1, 3, 4, 5) 
        attention_weights = attention_weights.permute(2, 0, 1, 3, 4, 5) 

        outputs = []
        for i, group in enumerate(groups):
            output = self.dilate_attention[i](group, sampling_locations[i], attention_weights[i], input_spatial_shapes)
            outputs.append(output)  
        output = torch.cat(outputs, dim=2) 

        output = self.out(output)
        return output

            
class DDAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, _d_per_head, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self._d_per_head = _d_per_head
        self.scale = qk_scale or _d_per_head ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, v, sampling_locations, attention_weights, input_spatial_shapes):

        b, h, c, lin = v.shape               
        _, lq, h, l, p = sampling_locations.shape    
        v = v.permute(0, 3, 1, 2) 
        attention_weights_s = attention_weights.reshape(b, lq, h, -1)   

        value_list = v.split([H_ * W_ for H_, W_ in input_spatial_shapes], dim=1) 
        sampling_grids = 2 * sampling_locations - 1                      #在[-1,1]之间
        sampling_value_list = []


        for lel, (H_, W_) in enumerate(input_spatial_shapes):
            value_h_l = value_list[lel].permute(0, 2, 3, 1).reshape(b*h, c, H_, W_)
            sampling_grid_l = sampling_grids[:, :, :, lel, :].permute(0, 2, 1, 3).reshape(b*h, lq, 1, p)   
            sampling_value_l = F.grid_sample(value_h_l, sampling_grid_l,
                                                  mode='bilinear', padding_mode='zeros', align_corners=False)        

            unfold_value_l = self.unfold(sampling_value_l).reshape([b*h, c, self.kernel_size * self.kernel_size, -1]).permute(0, 3, 1, 2) 
            sampling_value_list.append(unfold_value_l)
            
        sampling_value = torch.stack(sampling_value_list, dim=-1).reshape(b*h, lq, c, -1).permute(0, 1, 3, 2)   
        attn = attention_weights_s.permute(0, 2, 1, 3).reshape(b*h, lq, 1, -1)      
        output = (attn @ sampling_value).reshape(b, h, lq, c).permute(0, 2, 1, 3).reshape(b, lq, h*c)      
        return output


    
    
    


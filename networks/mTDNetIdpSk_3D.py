import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #当前程序上上一级目录
sys.path.append(BASE_DIR) #添加环境变量
from networks.networks_other import init_weights
from torch.distributions.uniform import Uniform

class ConvBlock3d(nn.Module): #dropout+dilation，其实和util文件里的UnetConv3_dropout一样
    """
    is_batchnorm:是否使用IN进行归一化
    结果: conv3d+(IN)+ReLU+[dropout]+Conv3d+(IN)+ReLU
    """
    def __init__(self, in_size, out_size, dropout_p, is_batchnorm, kernel_size=(3,3,3), \
        padding_size=(1,1,1), stride=(1,1,1), dilation = (1,1,1), init_type = 'kaiming'):# d x h x w
        super(ConvBlock3d, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, stride, padding_size, dilation),
                                       nn.InstanceNorm3d(out_size, affine=True),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, stride, padding_size, dilation),
                                       nn.InstanceNorm3d(out_size, affine=True),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, stride, padding_size, dilation),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, stride, padding_size, dilation),
                                       nn.ReLU(inplace=True),)

        self.dropout = nn.Dropout(dropout_p)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type = init_type)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.dropout(outputs)
        outputs = self.conv2(outputs)
        return outputs

class Unet_Upblock3d(nn.Module):#加了dropout
    """upsample +'compare shape' + cat + conv(含droupout); out = conv"""
    def __init__(self, in_size, out_size, dropout_p,  is_batchnorm=True, kernel_size=(3,3,3), padding_size = (1,1,1),\
         stride = (1,1,1), dilation= (1,1,1), init_type = 'kaiming'):
        super(Unet_Upblock3d, self).__init__()
        self.conv = ConvBlock3d(in_size + out_size, out_size, dropout_p, is_batchnorm, kernel_size, 
                                padding_size, stride, dilation) 
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear') #scale_factor是因为maxpool

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('ConvBlock3d') != -1: continue #查看每个submodule的类名，当前类名中有xxx的话，就continue
            init_weights(m, init_type = init_type)

    def forward(self, inputs1, inputs2):#inputs1是跳跃连接，inputs2是上采样
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]#这里是害怕up后的inputs2与input1的DxHxW不一致
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)#reshape 跳跃连接的output
        if np.shape(outputs1)!=np.shape(outputs2):#outputs2是decoder上采样的结果，outputs1是encoder下采样的结果
            _,_,d,h,w = outputs2.shape
            outputs1 = F.interpolate(outputs1, size=(d, h, w), mode='trilinear', align_corners=True)    
        x = torch.cat([outputs1, outputs2], 1)
        return self.conv(x)

class Encoder(nn.Module): 
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_channels = self.params['in_chns'] 
        self.feature_scale = self.params['feature_scale']        
        self.is_batchnorm = self.params['is_batchnorm'] # 默认True,IN
        self.dropout_p = self.params['dropout'] #[0.05, 0.1, 0.2, 0.3, 0.5]
        self.init_type = self.params['init_type']  #select mode: kaiming (default), xavier, normal, orthogonal
    

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        self.conv1 = ConvBlock3d(self.in_channels, filters[0], self.dropout_p[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = ConvBlock3d(filters[0], filters[1], self.dropout_p[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = ConvBlock3d(filters[1], filters[2], self.dropout_p[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = ConvBlock3d(filters[2], filters[3], self.dropout_p[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = ConvBlock3d(filters[3], filters[4], self.dropout_p[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # initialise weights 
        for m in self.modules(): #kaiming initialization
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type = self.init_type)
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type = self.init_type)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        
        maxpool4 = self.maxpool4(conv4)
        center = self.center(maxpool4)

        return [conv1, conv2, conv3, conv4, center]

def Dropout_random(x):
    dropout_p = round(random.uniform(0.2, 0.5),2)
    x = torch.nn.functional.dropout3d(x, dropout_p)
    return x 


def FeatureDropout_3D(x): 
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9) 
    threshold = threshold.view(x.size(0), 1, 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float() 
    x = x.mul(drop_mask)
    return x


######################################################################
## for conv1x1, denoted as m       
class Decoder_dilDrop_dp_multiscale(nn.Module): 
    def __init__(self, params):
        super(Decoder_dilDrop_dp_multiscale, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.n_class = self.params['class_num']
        self.feature_scale = self.params['feature_scale']
        self.is_batchnorm = self.params['is_batchnorm']
        self.init_type = self.params['init_type'] #select mode: kaiming, xavier, normal, orthogonal。默认kaiming
        self.padding_size = self.params['padding_size'] #3,1,6
        self.dilation = self.params['dilation'] #3,1,6
        self.dropout_p = self.params['dropout'] #[0.05, 0.1, 0.2, 0.3, 0.5]

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        self.up_concat4 = Unet_Upblock3d(filters[4], filters[3], self.dropout_p[3], self.is_batchnorm, padding_size = self.padding_size, \
            dilation = self.dilation) #UnetUp3_CT: up + cat + conv
        self.up_concat3 = Unet_Upblock3d(filters[3], filters[2], self.dropout_p[2], self.is_batchnorm, padding_size = self.padding_size, \
            dilation = self.dilation)
        self.up_concat2 = Unet_Upblock3d(filters[2], filters[1], self.dropout_p[1], self.is_batchnorm, padding_size = self.padding_size, \
            dilation = self.dilation)
        self.up_concat1 = Unet_Upblock3d(filters[1], filters[0], self.dropout_p[0], self.is_batchnorm, padding_size = self.padding_size, \
            dilation = self.dilation) # filters[1],[0]分别代表in, out,

        self.final = nn.Conv3d(filters[0], self.n_class, 1)#这一行来自unet3d
        self.out_conv1 = nn.Conv3d(filters[1], self.n_class, 1)
        self.out_conv2 = nn.Conv3d(filters[2], self.n_class, 1)
        self.out_conv3 = nn.Conv3d(filters[3], self.n_class, 1)
        # self.out_conv4 = nn.Conv3d(filters[4], self.n_class, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type=self.init_type)
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type=self.init_type)

    def forward(self, feature):
        conv1 = feature[0]
        conv2 = feature[1]
        conv3 = feature[2]
        conv4 = feature[3]
        center = feature[4]

        up4 = self.up_concat4(conv4, center) # 128
        out_4 = self.out_conv3(up4)
        up3 = self.up_concat3(conv3, up4) # 64
        out_3 = self.out_conv2(up3)
        up2 = self.up_concat2(conv2, up3) # 32
        out_2 = self.out_conv1(up2)
        up1 = self.up_concat1(conv1, up2) # 16
        output = self.final(up1) # class_num

        return output, out_2, out_3, out_4

class mTDNetIdpSk_3D(nn.Module): # m means using conv1x1,Sk means using same kernel
    # def __init__(self, in_channels=1, feature_scale=4, n_classes=8, 
    #              is_batchnorm=True, is_decoderRandomDropout=True, 
    #              is_sameInit = False, is_deepSupervision = False, same_kernel_size=1):
    
    def __init__(self, params):
        super(mTDNetIdpSk_3D, self).__init__()
        
        in_channels = params['in_chns']
        n_classes = params['class_num']
        feature_scale = params.get('feature_scale', 4)
        is_batchnorm = params.get('is_batchnorm', True)
        same_kernel_size = params.get('same_kernel_size', 1)
        is_sameInit = params.get('is_sameInit', False)
        is_decoderRandomDropout = params.get('is_decoderRandomDropout', True)
        is_deepSupervision = params.get('is_deepSupervision', True)

        params = {  'in_chns': in_channels,
                    'class_num': n_classes,
                    'feature_scale': feature_scale, # take place of original feature_channels:[16, 32, 64, 128, 256],
                    'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                    'is_batchnorm': is_batchnorm,
                    'init_type': 'kaiming', #select mode: kaiming, xavier, normal, orthogonal
                    'padding_size': same_kernel_size, 
                    'dilation': same_kernel_size} 
        params_deaux1 = {  'in_chns': in_channels,
                    'class_num': n_classes,
                    'feature_scale': feature_scale, 
                    'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                    'is_batchnorm': is_batchnorm,
                    'init_type': 'xavier', 
                    'padding_size':same_kernel_size, 
                    'dilation':same_kernel_size} 
        params_deaux2 = {  'in_chns': in_channels,
                    'class_num': n_classes,
                    'feature_scale': feature_scale, 
                    'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                    'is_batchnorm': is_batchnorm,
                    'init_type': 'normal', 
                    'padding_size':same_kernel_size, 
                    'dilation':same_kernel_size}  
        if is_sameInit:
            params_deaux1['init_type'] = 'kaiming'
            params_deaux2['init_type'] = 'kaiming'

        self.is_decoderRandomDropout = is_decoderRandomDropout
        self.is_deepSupervision = is_deepSupervision
        self.encoder = Encoder(params)
        self.main_decoder = Decoder_dilDrop_dp_multiscale(params)
        self.aux_decoder1 = Decoder_dilDrop_dp_multiscale(params_deaux1) 
        self.aux_decoder2 = Decoder_dilDrop_dp_multiscale(params_deaux2) 
        
    def forward(self, x): 
        # pLS论文里提到：
        # We added the dropout layer (ratio=0.5) before each
        # conv-block of the auxiliary decoder to introduce perturbations
        """
        here, we added random dropout before each conv-block of the auxiliary decoder"""
        features = self.encoder(x)
        main_seg, main_embedding1, main_embedding2, main_embedding3 = self.main_decoder(features)

        if not self.is_decoderRandomDropout:
            # aux1_feature = features
            # aux2_feature = features
            aux1_feature = [Dropout_random(i) for i in features]
            aux2_feature = [Dropout_random(i) for i in features]
        else:
            aux1_feature = [Dropout_random(i) for i in features]
            aux2_feature = [FeatureDropout_3D(i) for i in features]
      
        aux1_seg, aux1_embedding1, aux1_embedding2, aux1_embedding3 = self.aux_decoder1(aux1_feature)      
        aux2_seg, aux2_embedding1, aux2_embedding2, aux2_embedding3 = self.aux_decoder2(aux2_feature)
        if self.is_deepSupervision: 
            return [main_seg, aux1_seg, aux2_seg], [main_embedding1, aux1_embedding1, aux2_embedding1],\
               [main_embedding2, aux1_embedding2, aux2_embedding2], [main_embedding3, aux1_embedding3, aux2_embedding3]
        else: 
            return main_seg, aux1_seg, aux2_seg
    
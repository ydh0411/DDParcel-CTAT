
# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# =========================================================================
# networks.py — DDParcel 网络架构定义
# =========================================================================
# 本文件包含一系列网络变体，它们是在 FastSurferCNN（基础 U-Net）基础上
# 逐步演化而来的融合网络。DDParcel 推理实际使用的是：
#   FastSurferCNN_Fuse_Unet_v3_extended
#
# 网络变体家族：
# ┌─────────────────────────────────────────────────────────────────────┐
# │ FastSurferCNN                    ✓ 基础 U-Net，单模态输入           │
# │ FastSurferCNN_return_all         ✓ 同左，但返回所有中间特征         │
# │ FastSurferCNN_no_classifer       ✓ 无分类头的特征提取器            │
# │ FastSurferCNN_Fuse_Last_Layer    ★ 多模态，仅在最后融合            │
# │ FastSurferCNN_Fuse_Unet          ★ 多模态，全编码器层竞争融合      │
# │ FastSurferCNN_Fuse_Unet_v1       ★ v1：带 backbone 辅助输出加权   │
# │ FastSurferCNN_Fuse_Unet_v2       ★ v2：模态拼接输入 + 竞争融合    │
# │ FastSurferCNN_Fuse_Unet_v2_ext   ★ v2 extended：返回多路 logits   │
# │ FastSurferCNN_Fuse_Unet_v3       ★ v3：拼接解码特征再分类         │
# │ FastSurferCNN_Fuse_Unet_v3_ext ★★★ v3 extended：最终版本 ✓ 使用了 │
# │ FastSurferCNN_Fuse_Unet_v4       ★ v4：类似 v3，返回更多辅助信息  │
# │ FastSurferCNN_Fuse_Unet_v4_ext   ★ v4 extended                   │
# └─────────────────────────────────────────────────────────────────────┘
#
# === 设计演进的直观理解 ===
# 原始 FastSurferCNN 是一个标准 U-Net，输入 7 通道（厚切片）。
# 到了 DDParcel，我们需要同时输入 FA、Trace、MinEig、MidEig 四个 DTI 模态。
# 最简单的想法是「通道拼接」（v2），但效果不好。
# 更好的思路是「各模态先独立提取特征，再融合」（v3 和 v3_extended）。
# 这类似多专家系统：每个模态有一个「专家 backbone」，MAE 模块在各层做 maxout 竞争融合，最后再分类。 
# 融合分支学习如何组合他们的意见。
# =========================================================================
import torch.nn as nn
import torch
import models.sub_module as sm#models.sub_module 包含了 FastSurferCNN 的各个模块定义，如 CompetitiveEncoderBlock、CompetitiveDecoderBlock、ClassifierBlock 等。
from collections import OrderedDict


# =========================================================================
# 1. FastSurferCNN — 基础 U-Net（单模态）
# =========================================================================
# 这是整个家族的「根」。标准 U-Net 结构：
#
#  输入 (C×H×W)
#      ↓
#  ┌──────────────────┐
#  │ Encoder Block 1  │ → 跳跃连接 (skip_1)
#  │ MaxPool          │ → 128×128
#  ├──────────────────┤
#  │ Encoder Block 2  │ → 跳跃连接 (skip_2)
#  │ MaxPool          │ → 64×64
#  ├──────────────────┤
#  │ Encoder Block 3  │ → 跳跃连接 (skip_3)
#  │ MaxPool          │ → 32×32
#  ├──────────────────┤
#  │ Encoder Block 4  │ → 跳跃连接 (skip_4)
#  │ MaxPool          │ → 16×16
#  ├──────────────────┤
#  │  Bottleneck      │ → 最深层特征
#  ├──────────────────┤
#  │ Decoder Block 4  │ ← + skip_4 (Unpool → 32×32)
#  ├──────────────────┤
#  │ Decoder Block 3  │ ← + skip_3 (Unpool → 64×64)
#  ├──────────────────┤
#  │ Decoder Block 2  │ ← + skip_2 (Unpool → 128×128)
#  ├──────────────────┤
#  │ Decoder Block 1  │ ← + skip_1 (Unpool → 256×256)
#  ├──────────────────┤
#  │  Classifier      │ → 1×1 conv → C_class 通道 logits
#  └──────────────────┘
# =========================================================================
class FastSurferCNN(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params):
        super(FastSurferCNN, self).__init__()

        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        # CompetitiveEncoderBlockInput 是编码器第一层的特殊版本，输入通道数是 params['num_channels']，输出通道数是 params['num_filters']。
        #CompetitiveEncoderBlockInput是一个利用competitive dense block(CDB)的编码器块，利用maxout的竞争机制来融合卷积块的响应更强的特征，而第一层
        #第一层结构batchnorm+conv+batchnorm+后续的competitive dense block结构相同，区别在于输入通道数和输出通道数的设置。
        #后面的CompetitiveEncoderBlock是编码器的标准版本，输入通道数和输出通道数都是 params['num_filters']。
        #结构是：PReLU + Conv + BatchNorm + Competitive Dense Block，Competitive Dense Block 内部有多个卷积层，每层的输入是前面所有层的输出的拼接，输出通道数是 params['num_filters']。
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)
        #bottleneck 是编码器和解码器之间的连接层，结构是一个 Competitive Dense Block，输入通道数和输出通道数都是 params['num_filters']。
        #因为前面的encode过程会导致空间尺寸变小，但通道数增加，可以理解为把前面提取的特征进行整合，抓住最核心的特征，作为decoder的输入。

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        #
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)
        #decoder部分的 CompetitiveDecoderBlock 结构是：
        # Unpool + PReLU + Conv + BatchNorm + Competitive Dense Block，输入通道数和输出通道数都是 params['num_filters']。
        #Unpool 是反池化操作，利用前面编码器的池化索引来恢复空间尺寸，跳跃连接（skip）是来自编码器对应层的特征图，通过 maxout 竞争融合后输入到 Competitive Dense Block。

        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)
        #classifier 是最后的分类层，结构是：PReLU + Conv (kernel_size=1) + BatchNorm，输入通道数是 params['num_filters']，输出通道数是 params['num_classes']，即每个类别对应一个通道的 logits。

        # Code for Network Initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                #kaiming_normal_ 是一种权重初始化方法，适用于 ReLU 或 Leaky ReLU 激活函数的网络层。它根据输入通道数自动计算合适的标准差来初始化卷积层的权重，使得前向传播时信号能够保持稳定。
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)#BatchNorm 层的权重初始化为 1，偏置初始化为 0，这样在训练开始时 BatchNorm 不会改变输入的分布。
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """
        #skip connection 和 indices 是为了 Unpooling 操作准备的，Unpooling 需要知道之前池化的位置和索引来恢复空间尺寸。
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x)
        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)
        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)
        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)

        bottleneck = self.bottleneck(encoder_output4)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)

        logits = self.classifier.forward(decoder_output1)

        return logits


# =========================================================================
# 1b. FastSurferCNN_return_all — 同基础 U-Net，但返回所有中间特征
# =========================================================================
# 和基础 FastSurferCNN 结构完全相同，唯一区别是 forward 的返回值。
# 它返回编码器/解码器每一层的输出、跳跃连接和池化索引。
#
# 为什么要保留所有中间特征？
# 在多模态融合版本（如 v3_extended）中，每个模态的 backbone 用的是这个类。
# 融合分支需要读取每个 backbone 在各层（编码 1~4、bottleneck、解码 1~4）的输出，
# 来做跨模态的 maxout 竞争融合。
#
# 所以这个类相当于「会分享中间结果的特征提取器」。
# =========================================================================
class FastSurferCNN_return_all(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    #这个版本的fastsurfercnn和基础版本结构完全一样，区别在于forward函数的返回值。它不仅返回最终的logits，
    # 还返回了编码器和解码器每一层的输出、跳跃连接和池化索引。这些中间特征对于后续的多模态融合非常重要，
    # 因为多模态是需要多种类别的特征，因为融合分支需要读取这些特征来进行跨模态的 maxout 竞争融合。
    def __init__(self, params):
        super(FastSurferCNN_return_all, self).__init__()

        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)

        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits, plus ALL intermediate feature maps
        """
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x)
        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)
        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)
        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)

        bottleneck = self.bottleneck(encoder_output4)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)

        logits = self.classifier.forward(decoder_output1)

        # ★ 返回 1 个主结果 + 18 个中间量
        # 索引对应关系：
        # [0]  logits
        # [1]  encoder_output1,  [2]  skip_encoder_1,  [3]  indices_1
        # [4]  encoder_output2,  [5]  skip_encoder_2,  [6]  indices_2
        # [7]  encoder_output3,  [8]  skip_encoder_3,  [9]  indices_3
        # [10] encoder_output4,  [11] skip_encoder_4,  [12] indices_4
        # [13] bottleneck
        # [14] decoder_output4, [15] decoder_output3, [16] decoder_output2, [17] decoder_output1
        return logits, \
               encoder_output1, skip_encoder_1, indices_1, \
               encoder_output2, skip_encoder_2, indices_2, \
               encoder_output3, skip_encoder_3, indices_3, \
               encoder_output4, skip_encoder_4, indices_4, \
               bottleneck, \
               decoder_output4, decoder_output3, decoder_output2, decoder_output1


# =========================================================================
# 1c. FastSurferCNN_no_classifer — 无分类头的特征提取器
# =========================================================================
# 去掉分类器（1x1 卷积映射到类别数），只输出解码器最后一层的特征。
# 在其他融合网络（如 Fuse_Last_Layer）中被用作 backbone。
# =========================================================================
class FastSurferCNN_no_classifer(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params):
        super(FastSurferCNN_no_classifer, self).__init__()

        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)

        # Code for Network Initialization
#这里没有分类头了，所以不需要初始化分类层的权重了，其他部分和基础版本一样。
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x)
        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)
        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)
        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)

        bottleneck = self.bottleneck(encoder_output4)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)

        return decoder_output1


# =========================================================================
# 2. FastSurferCNN_Fuse_Last_Layer — 早期融合实验：仅在最后融合
# =========================================================================
# 思路：每个模态独立通过无分类头的 U-Net，得到解码特征后，
# 将所有模态的特征拼接起来，通过一个融合卷积，再做分类。
#
# 这相当于「各模态各走各的路，最后才见面」的策略。
# 缺点是缺少中间层的跨模态交互。
# =========================================================================
class FastSurferCNN_Fuse_Last_Layer(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params):
        super(FastSurferCNN_Fuse_Last_Layer, self).__init__()

        num_channels = params['num_channels']
        # Parameters for the Descending Arm
        self.FastSurferCNNs = nn.ModuleList()#这个是直接调用上面定义的 FastSurferCNN_no_classifer 类，作为每个模态的 backbone。每个 backbone 都是一个独立的 U-Net，输入是对应模态的 7 通道切片，输出是解码器最后一层的特征图。
        for idx in range(params['num_modality']):#这里的 params['num_modality'] 是模态数量，比如 4（FA、Trace、MinEig、MidEig）。循环中每次创建一个 FastSurferCNN_no_classifer 实例，作为一个模态的特征提取器。
            params['num_channels'] = num_channels
            self.FastSurferCNNs.append(FastSurferCNN_no_classifer(params))#这里的 FastSurferCNN_no_classifer 是上面定义的那个没有分类头的版本，输出是解码器最后一层的特征图，通道数是 params['num_filters']。
            #append了什么？append了一个 FastSurferCNN_no_classifer 的实例，这个实例是一个 U-Net 结构的特征提取器，输入是对应模态的 7 通道切片，输出是解码器最后一层的特征图，通道数是 params['num_filters']。

        self.fusion_layer = nn.Conv2d(params['num_filters'] * params['num_modality'], params['num_filters'], params['kernel_c'], params['stride_conv'])  # To generate logits

        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

    def forward(self, x):
        decoder_outputs = []
        for idx, fscnn in enumerate(self.FastSurferCNNs):
            # 每个模态取自己的 7 通道切片 → 独立的 U-Net 前向
            decoder_outputs.append(fscnn(x[:, (idx*7):((idx+1) * 7), :, :]))

        # 所有模态的解码特征拼接 → 1x1 融合 → 分类
        decoder_outputs_fused = self.fusion_layer(torch.cat(decoder_outputs, dim=1))

        logits = self.classifier.forward(decoder_outputs_fused)

        return logits


# =========================================================================
# 3. FastSurferCNN_Fuse_Unet — 深层融合：编码器各阶段做融合
# =========================================================================
# 改进 Last_Layer 版本：在编码器的每个阶段（encoder1~4）和解码前（bottleneck）
# 都用 maxout 竞争融合多个模态的特征，然后再继续前向。
#
# 这相当于在每个分辨率层级上都做跨模态信息交换。
# =========================================================================
class FastSurferCNN_Fuse_Unet(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params):
        super(FastSurferCNN_Fuse_Unet, self).__init__()

        num_channels = params['num_channels']

        # Load backbones
        self.FastSurferCNNs = nn.ModuleList()
        for idx in range(params['num_modality']):
            params['num_channels'] = num_channels
            backbone_model = FastSurferCNN_return_all(params)
            if params['backbone_model'] is not None:
                model_state = torch.load(params['backbone_model'][idx])
                new_state_dict = OrderedDict()
                for k, v in model_state["model_state_dict"].items():
                    if k[:7] == "module.":
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                backbone_model.load_state_dict(new_state_dict)
                for param in backbone_model.parameters():
                    param.requires_grad = False
            self.FastSurferCNNs.append(backbone_model)

        params['num_channels'] = num_channels
        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)

        self.fusion1 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion2 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion3 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion4 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion5 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])

        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        returns = []
        for idx in range(len(self.FastSurferCNNs)):
            fscnn = self.FastSurferCNNs[idx]
            returns.append(fscnn(x[:, (idx*7):((idx+1) * 7), :, :]))

        idx = 0
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x[:, (idx*7):((idx+1) * 7), :, :])

        encoder_output1 = torch.unsqueeze(encoder_output1, 4)
        for idx in range(1, len(self.FastSurferCNNs)):
            encoder_output1 = torch.cat((encoder_output1, returns[idx][1].unsqueeze(4)), dim=4)
        encoder_output1, _ = torch.max(encoder_output1, 4)

        encoder_output1 = self.fusion1(encoder_output1)

        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)

        encoder_output2 = torch.unsqueeze(encoder_output2, 4)
        for idx in range(1, len(self.FastSurferCNNs)):
            encoder_output2 = torch.cat((encoder_output2, returns[idx][4].unsqueeze(4)), dim=4)
        encoder_output2, _ = torch.max(encoder_output2, 4)

        encoder_output2 = self.fusion2(encoder_output2)

        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)

        encoder_output3 = torch.unsqueeze(encoder_output3, 4)
        for idx in range(1, len(self.FastSurferCNNs)):
            encoder_output3 = torch.cat((encoder_output3, returns[idx][7].unsqueeze(4)), dim=4)
        encoder_output3, _ = torch.max(encoder_output3, 4)

        encoder_output3 = self.fusion3(encoder_output3)

        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)

        encoder_output4 = torch.unsqueeze(encoder_output4, 4)
        for idx in range(1, len(self.FastSurferCNNs)):
            encoder_output4 = torch.cat((encoder_output4, returns[idx][10].unsqueeze(4)), dim=4)
        encoder_output4, _ = torch.max(encoder_output4, 4)

        encoder_output4 = self.fusion4(encoder_output4)

        bottleneck = self.bottleneck(encoder_output4)

        bottleneck = torch.unsqueeze(bottleneck, 4)
        for idx in range(1, len(self.FastSurferCNNs)):
            bottleneck = torch.cat((bottleneck, returns[idx][13].unsqueeze(4)), dim=4)
        bottleneck, _ = torch.max(bottleneck, 4)

        bottleneck = self.fusion5(bottleneck)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)

        logits = self.classifier.forward(decoder_output1)

        return logits


# =========================================================================
# 4-7. v1 ~ v2_extended — 中间实验版本
# =========================================================================
# 这些版本尝试了不同融合策略，主要在以下几点有所不同：
# - 输入用主融合分支的哪个通道切片（idx*7 还是 (idx+1)*7）
# - 辅助 backbone 是否参与解码器的特征融合
# - 返回的 logits_list 是否包含各 backbone 的独立输出
#
# 由于 DDParcel 最终使用的是 v3_extended，这里只保留简要结构说明。
# 如果想深入理解设计演进，建议对照 v2 → v3 → v3_extended 的差异。
# =========================================================================

class FastSurferCNN_Fuse_Unet_v1(nn.Module):
    """v1：主分支处理模态 0，backbone 处理模态 1~N，最后加权相加 logits"""
    def __init__(self, params):
        super(FastSurferCNN_Fuse_Unet_v1, self).__init__()
        num_channels = params['num_channels']
        self.FastSurferCNNs = nn.ModuleList()
        for idx in range(params['num_modality']-1):
            params['num_channels'] = num_channels
            backbone_model = FastSurferCNN_return_all(params)
            if params['backbone_model'] is not None:
                print('Loading: %s' % params['backbone_model'][idx+1])
                model_state = torch.load(params['backbone_model'][idx+1])
                new_state_dict = OrderedDict()
                for k, v in model_state["model_state_dict"].items():
                    if k[:7] == "module.":
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                backbone_model.load_state_dict(new_state_dict)
                for param in backbone_model.parameters():
                    param.requires_grad = False
            self.FastSurferCNNs.append(backbone_model)
        params['num_channels'] = num_channels
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)
        self.fusion1 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion2 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion3 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion4 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion5 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        returns = []
        for idx in range(len(self.FastSurferCNNs)):
            fscnn = self.FastSurferCNNs[idx]
            returns.append(fscnn(x[:, ((idx+1)*7):((idx+2) * 7), :, :]))
        idx = 0
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x[:, (idx*7):((idx+1) * 7), :, :])
        encoder_output1 = torch.unsqueeze(encoder_output1, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output1 = torch.cat((encoder_output1, returns[idx][1].unsqueeze(4)), dim=4)
        encoder_output1, _ = torch.max(encoder_output1, 4)
        encoder_output1 = self.fusion1(encoder_output1)
        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)
        encoder_output2 = torch.unsqueeze(encoder_output2, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output2 = torch.cat((encoder_output2, returns[idx][4].unsqueeze(4)), dim=4)
        encoder_output2, _ = torch.max(encoder_output2, 4)
        encoder_output2 = self.fusion2(encoder_output2)
        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)
        encoder_output3 = torch.unsqueeze(encoder_output3, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output3 = torch.cat((encoder_output3, returns[idx][7].unsqueeze(4)), dim=4)
        encoder_output3, _ = torch.max(encoder_output3, 4)
        encoder_output3 = self.fusion3(encoder_output3)
        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)
        encoder_output4 = torch.unsqueeze(encoder_output4, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output4 = torch.cat((encoder_output4, returns[idx][10].unsqueeze(4)), dim=4)
        encoder_output4, _ = torch.max(encoder_output4, 4)
        encoder_output4 = self.fusion4(encoder_output4)
        bottleneck = self.bottleneck(encoder_output4)
        bottleneck = torch.unsqueeze(bottleneck, 4)
        for idx in range(len(self.FastSurferCNNs)):
            bottleneck = torch.cat((bottleneck, returns[idx][13].unsqueeze(4)), dim=4)
        bottleneck, _ = torch.max(bottleneck, 4)
        bottleneck = self.fusion5(bottleneck)
        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)
        logits = self.classifier.forward(decoder_output1)
        for ind_return in returns:
            logits += ind_return[0]
        return logits


class FastSurferCNN_Fuse_Unet_v2(nn.Module):
    """v2：主分支接收所有模态的拼接输入 (num_modality * 7 通道)，backbone 作辅助"""
    def __init__(self, params):
        super(FastSurferCNN_Fuse_Unet_v2, self).__init__()
        num_channels = params['num_channels']
        params['num_channels'] = params['num_modality'] * 7
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)
        self.fusion1 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion2 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion3 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion4 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion5 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.FastSurferCNNs = nn.ModuleList()
        for idx in range(params['num_modality']):
            params['num_channels'] = num_channels
            backbone_model = FastSurferCNN_return_all(params)
            if params['backbone_model'] is not None:
                print('Loading: %s' % params['backbone_model'][idx])
                model_state = torch.load(params['backbone_model'][idx])
                new_state_dict = OrderedDict()
                for k, v in model_state["model_state_dict"].items():
                    if k[:7] == "module.":
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                backbone_model.load_state_dict(new_state_dict)
                for param in backbone_model.parameters():
                    param.requires_grad = False
            else:
                print('No Backbone!!!')
            self.FastSurferCNNs.append(backbone_model)
        print('Initialize Fuse Unet v2 done!')

    def forward(self, x):
        returns = []
        for idx in range(len(self.FastSurferCNNs)):
            fscnn = self.FastSurferCNNs[idx]
            returns.append(fscnn(x[:, ((idx)*7):((idx+1) * 7), :, :]))
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x[:, :, :, :])
        encoder_output1 = torch.unsqueeze(encoder_output1, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output1 = torch.cat((encoder_output1, returns[idx][1].unsqueeze(4)), dim=4)
        encoder_output1, _ = torch.max(encoder_output1, 4)
        encoder_output1 = self.fusion1(encoder_output1)
        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)
        encoder_output2 = torch.unsqueeze(encoder_output2, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output2 = torch.cat((encoder_output2, returns[idx][4].unsqueeze(4)), dim=4)
        encoder_output2, _ = torch.max(encoder_output2, 4)
        encoder_output2 = self.fusion2(encoder_output2)
        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)
        encoder_output3 = torch.unsqueeze(encoder_output3, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output3 = torch.cat((encoder_output3, returns[idx][7].unsqueeze(4)), dim=4)
        encoder_output3, _ = torch.max(encoder_output3, 4)
        encoder_output3 = self.fusion3(encoder_output3)
        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)
        encoder_output4 = torch.unsqueeze(encoder_output4, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output4 = torch.cat((encoder_output4, returns[idx][10].unsqueeze(4)), dim=4)
        encoder_output4, _ = torch.max(encoder_output4, 4)
        encoder_output4 = self.fusion4(encoder_output4)
        bottleneck = self.bottleneck(encoder_output4)
        bottleneck = torch.unsqueeze(bottleneck, 4)
        for idx in range(len(self.FastSurferCNNs)):
            bottleneck = torch.cat((bottleneck, returns[idx][13].unsqueeze(4)), dim=4)
        bottleneck, _ = torch.max(bottleneck, 4)
        bottleneck = self.fusion5(bottleneck)
        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)
        logits_list = []
        logits = self.classifier.forward(decoder_output1)
        logits_list.append(logits)
        for ind_return in returns:
            logits += ind_return[0]
            logits_list.append(ind_return[0])
        return logits


class FastSurferCNN_Fuse_Unet_v2_extended(nn.Module):
    """v2 extended：与 v2 结构相同，但返回 (logits, logits_list) 元组"""
    def __init__(self, params):
        super(FastSurferCNN_Fuse_Unet_v2_extended, self).__init__()
        num_channels = params['num_channels']
        params['num_channels'] = params['num_modality'] * 7
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)
        self.fusion1 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion2 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion3 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion4 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion5 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.FastSurferCNNs = nn.ModuleList()
        for idx in range(params['num_modality']):
            params['num_channels'] = num_channels
            backbone_model = FastSurferCNN_return_all(params)
            if params['backbone_model'] is not None:
                print('Loading: %s' % params['backbone_model'][idx])
                model_state = torch.load(params['backbone_model'][idx])
                new_state_dict = OrderedDict()
                for k, v in model_state["model_state_dict"].items():
                    if k[:7] == "module.":
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                backbone_model.load_state_dict(new_state_dict)
                for param in backbone_model.parameters():
                    param.requires_grad = False
            self.FastSurferCNNs.append(backbone_model)

    def forward(self, x):
        returns = []
        for idx in range(len(self.FastSurferCNNs)):
            fscnn = self.FastSurferCNNs[idx]
            returns.append(fscnn(x[:, ((idx)*7):((idx+1) * 7), :, :]))
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x[:, :, :, :])
        encoder_output1 = torch.unsqueeze(encoder_output1, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output1 = torch.cat((encoder_output1, returns[idx][1].unsqueeze(4)), dim=4)
        encoder_output1, _ = torch.max(encoder_output1, 4)
        encoder_output1 = self.fusion1(encoder_output1)
        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)
        encoder_output2 = torch.unsqueeze(encoder_output2, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output2 = torch.cat((encoder_output2, returns[idx][4].unsqueeze(4)), dim=4)
        encoder_output2, _ = torch.max(encoder_output2, 4)
        encoder_output2 = self.fusion2(encoder_output2)
        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)
        encoder_output3 = torch.unsqueeze(encoder_output3, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output3 = torch.cat((encoder_output3, returns[idx][7].unsqueeze(4)), dim=4)
        encoder_output3, _ = torch.max(encoder_output3, 4)
        encoder_output3 = self.fusion3(encoder_output3)
        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)
        encoder_output4 = torch.unsqueeze(encoder_output4, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output4 = torch.cat((encoder_output4, returns[idx][10].unsqueeze(4)), dim=4)
        encoder_output4, _ = torch.max(encoder_output4, 4)
        encoder_output4 = self.fusion4(encoder_output4)
        bottleneck = self.bottleneck(encoder_output4)
        bottleneck = torch.unsqueeze(bottleneck, 4)
        for idx in range(len(self.FastSurferCNNs)):
            bottleneck = torch.cat((bottleneck, returns[idx][13].unsqueeze(4)), dim=4)
        bottleneck, _ = torch.max(bottleneck, 4)
        bottleneck = self.fusion5(bottleneck)
        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)
        logits_list = []
        logits = self.classifier.forward(decoder_output1)
        logits_list.append(logits)
        for ind_return in returns:
            logits += ind_return[0]
            logits_list.append(ind_return[0])
        return logits, logits_list


# =========================================================================
# 8-9. v3 / v3_extended — ★ DDParcel 实际使用的架构 ★
# =========================================================================
# 从 v3 开始，关键的架构改变是：
# 在最终分类前，显式融合「主融合分支解码器输出 + 各 backbone 解码器输出」，
# 然后再做分类。
#
# v3：返回 logits（主融合 + backbone 之和）
# v3_extended（★★★ 实际使用）：返回 (logits, logits_list)
#   - logits[0] = 主融合分支的分类结果
#   - logits_list = [主融合结果, backbone0结果, backbone1结果, ...]
#   推理脚本中用 logits_list 做多路概率累加（权重 0.4/0.4/0.2）
# =========================================================================

class FastSurferCNN_Fuse_Unet_v3(nn.Module):
    """v3：主分支接收拼接输入，同时在编码各阶段做 maxout 竞争，
    最终在解码后融合所有分支的解码特征再分类"""
    def __init__(self, params):
        super(FastSurferCNN_Fuse_Unet_v3, self).__init__()
        num_channels = params['num_channels']
        params['num_channels'] = params['num_modality'] * 7
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)
        self.fusion1 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion2 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion3 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion4 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion5 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        # ★ v3 新增：融合层通道数 = num_filters * (num_modality + 1)
        # 其中 +1 是主融合分支，num_modality 是各 backbone 的解码输出
        self.fusion_layer = nn.Conv2d(params['num_filters'] * (params['num_modality'] + 1), params['num_filters'], params['kernel_c'], params['stride_conv'])
        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.FastSurferCNNs = nn.ModuleList()
        for idx in range(params['num_modality']):
            params['num_channels'] = num_channels
            backbone_model = FastSurferCNN_return_all(params)
            if params['backbone_model'] is not None:
                print('Loading: %s' % params['backbone_model'][idx])
                model_state = torch.load(params['backbone_model'][idx])
                new_state_dict = OrderedDict()
                for k, v in model_state["model_state_dict"].items():
                    if k[:7] == "module.":
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                backbone_model.load_state_dict(new_state_dict)
                for param in backbone_model.parameters():
                    param.requires_grad = False
            else:
                print('No Backbone!!!')
            self.FastSurferCNNs.append(backbone_model)
        print('Initialize Fuse Unet v3 done!')

    def forward(self, x):
        returns = []
        for idx in range(len(self.FastSurferCNNs)):
            fscnn = self.FastSurferCNNs[idx]
            returns.append(fscnn(x[:, ((idx)*7):((idx+1) * 7), :, :]))
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x[:, :, :, :])
        encoder_output1 = torch.unsqueeze(encoder_output1, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output1 = torch.cat((encoder_output1, returns[idx][1].unsqueeze(4)), dim=4)
        encoder_output1, _ = torch.max(encoder_output1, 4)
        encoder_output1 = self.fusion1(encoder_output1)
        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)
        encoder_output2 = torch.unsqueeze(encoder_output2, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output2 = torch.cat((encoder_output2, returns[idx][4].unsqueeze(4)), dim=4)
        encoder_output2, _ = torch.max(encoder_output2, 4)
        encoder_output2 = self.fusion2(encoder_output2)
        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)
        encoder_output3 = torch.unsqueeze(encoder_output3, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output3 = torch.cat((encoder_output3, returns[idx][7].unsqueeze(4)), dim=4)
        encoder_output3, _ = torch.max(encoder_output3, 4)
        encoder_output3 = self.fusion3(encoder_output3)
        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)
        encoder_output4 = torch.unsqueeze(encoder_output4, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output4 = torch.cat((encoder_output4, returns[idx][10].unsqueeze(4)), dim=4)
        encoder_output4, _ = torch.max(encoder_output4, 4)
        encoder_output4 = self.fusion4(encoder_output4)
        bottleneck = self.bottleneck(encoder_output4)
        bottleneck = torch.unsqueeze(bottleneck, 4)
        for idx in range(len(self.FastSurferCNNs)):
            bottleneck = torch.cat((bottleneck, returns[idx][13].unsqueeze(4)), dim=4)
        bottleneck, _ = torch.max(bottleneck, 4)
        bottleneck = self.fusion5(bottleneck)
        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)
        # ★ 融合多路解码特征：主分支(decoder_output1) + 各 backbone 的解码输出
        decoder_outputs = []
        decoder_outputs.append(decoder_output1)
        for ind_return in returns:
            decoder_outputs.append(ind_return[17])  # decoder_output1 of each backbone
        decoder_outputs_fused = self.fusion_layer(torch.cat(decoder_outputs, dim=1))
        logits = self.classifier.forward(decoder_outputs_fused)
        logits_list = []
        logits_list.append(logits)
        for ind_return in returns:
            logits += ind_return[0]
            logits_list.append(ind_return[0])
        return logits


# =========================================================================
# ★★★ FastSurferCNN_Fuse_Unet_v3_extended — DDParcel 推理使用 ★★★
# =========================================================================
# 这是 DDParcel 推理脚本 (DDSurfer_Pred.py) 实际调用的网络类。
# 和 v3 的区别：返回 (logits, logits_list) 供外部多路概率累加。
#
# 整体架构图（建议反复对照）：
#
#  输入 x：[B, 28, H, W]（4 模态 × 7 厚切片 = 28 通道）
#      │
#      ├─[模态 0: ch 0-7]──→ backbone0 → 各层特征 [skip1...decoder1]
#      ├─[模态 1: ch 7-14]─→ backbone1 → 各层特征
#      ├─[模态 2: ch 14-21]→ backbone2 → 各层特征
#      └─[模态 3: ch 21-28]→ backbone3 → 各层特征
#      │                              (所有 backbone 参数冻结，只做特征提取)
#      │
#      └─[全部 28 通道]──→ 主融合分支
#               │
#          ┌────┴────┐
#          │encode1  │ ←── maxout(主分支特征, backbone0_e1, backbone1_e1, ...) → fusion1
#          │encode2  │ ←── maxout(主分支特征, backbone0_e2, backbone1_e2, ...) → fusion2
#          │encode3  │ ←── maxout(...) → fusion3
#          │encode4  │ ←── maxout(...) → fusion4
#          │bottleneck│ ←── maxout(...) → fusion5
#          │decode4~1 │
#          └────┬────┘
#               │ decoder_output1（主融合解码特征）
#               │
#          ┌────┴─────────────────────┐
#          │ concat(主融合, bbone0_d1, │
#          │        bbone1_d1, ...)   │ ← ★ 这是 v3 的核心改进
#          └────────┬─────────────────┘
#                   ↓ fusion_layer(1×1 conv)
#                   ↓ classifier(1×1 conv)
#                   ↓
#              logits（主输出）
#              logits_list（含各 backbone 独立 logits 供累加）
# =========================================================================
class FastSurferCNN_Fuse_Unet_v3_extended(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params):
        super(FastSurferCNN_Fuse_Unet_v3_extended, self).__init__()

        num_channels = params['num_channels']

        # 主融合编码-解码分支接收所有模态拼接后的输入
        #（每个模态贡献 7 个厚切片通道）。
        # 例如：4 模态时，输入通道数 = 4 * 7 = 28。
        params['num_channels'] = params['num_modality'] * 7
        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)

        # 5 个融合卷积（每个编码阶段 + bottleneck 各一个）
        # 输入输出都是 num_filters 通道，kernel 尺寸通常为 1
        self.fusion1 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion2 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion3 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion4 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion5 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])

        # 最终融合层混合以下解码器输出：
        # - 主融合分支（1 路）
        # - 各模态 backbone 分支（num_modality 路）
        # 因此输入通道 = num_filters * (num_modality + 1)。
        self.fusion_layer = nn.Conv2d(params['num_filters'] * (params['num_modality'] + 1), params['num_filters'], params['kernel_c'], params['stride_conv'])

        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 加载每个模态的预训练 backbone，并冻结参数作为特征专家。
        # 直观上：每个 backbone 只精通一种模态（如 FA/Trace/...）。
        self.FastSurferCNNs = nn.ModuleList()
        for idx in range(params['num_modality']):
            params['num_channels'] = num_channels
            backbone_model = FastSurferCNN_return_all(params)
            if params['backbone_model'] is not None:
                print('Loading: %s' % params['backbone_model'][idx])
                if params["use_cuda"]:
                    model_state = torch.load(params['backbone_model'][idx])
                else:
                    model_state = torch.load(params['backbone_model'][idx], map_location=torch.device('cpu'))
                new_state_dict = OrderedDict()
                for k, v in model_state["model_state_dict"].items():
                    if k[:7] == "module.":
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                backbone_model.load_state_dict(new_state_dict)
                # ★ 冻结 backbone 参数：只做推理，不参与训练
                for param in backbone_model.parameters():
                    param.requires_grad = False
            else:
                print('No Backbone!!!')
            self.FastSurferCNNs.append(backbone_model)

        print('Initialize Fuse Unet v3 done!')

    def forward(self, x):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """

        # return [0] logits, \
        #        [1] encoder_output1,  [2] skip_encoder_1,  [3] indices_1, \
        #        [4] encoder_output2,  [5] skip_encoder_2,  [6] indices_2, \
        #        [7] encoder_output3,  [8] skip_encoder_3,  [9] indices_3, \
        #        [10] encoder_output4, [11] skip_encoder_4, [12] indices_4, \
        #        [13] bottleneck, \
        #        [14] decoder_output4, [15] decoder_output3, [16] decoder_output2, [17] decoder_output1

        # =====================================================================
        # Step 1: 各模态 backbone 前向
        # =====================================================================
        # 每个模态 backbone 只处理自己对应的 7 通道输入块。
        # x 的通道切片方式：
        #   模态0 -> x[:, 0:7, ...]   (FA)
        #   模态1 -> x[:, 7:14, ...]  (Trace)
        #   模态2 -> x[:, 14:21, ...] (MinEig)
        #   模态3 -> x[:, 21:28, ...] (MidEig)
        # 每个 returns[idx] 包含该 backbone 所有中间层输出
        returns = []
        for idx in range(len(self.FastSurferCNNs)):
            fscnn = self.FastSurferCNNs[idx]
            returns.append(fscnn(x[:, ((idx)*7):((idx+1) * 7), :, :]))

        # =====================================================================
        # Step 2-6: 主融合分支 + 跨模态竞争融合
        # =====================================================================
        # 融合分支：每个编码阶段先在多模态分支间做 max 竞争，
        # 再通过 1x1 卷积进行融合细化。
        # 可以理解为「先选最强，再学习如何混合」。
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x[:, :, :, :])

        encoder_output1 = torch.unsqueeze(encoder_output1, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output1 = torch.cat((encoder_output1, returns[idx][1].unsqueeze(4)), dim=4)
        encoder_output1, _ = torch.max(encoder_output1, 4)
        encoder_output1 = self.fusion1(encoder_output1)

        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)

        encoder_output2 = torch.unsqueeze(encoder_output2, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output2 = torch.cat((encoder_output2, returns[idx][4].unsqueeze(4)), dim=4)
        encoder_output2, _ = torch.max(encoder_output2, 4)
        encoder_output2 = self.fusion2(encoder_output2)

        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)

        encoder_output3 = torch.unsqueeze(encoder_output3, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output3 = torch.cat((encoder_output3, returns[idx][7].unsqueeze(4)), dim=4)
        encoder_output3, _ = torch.max(encoder_output3, 4)
        encoder_output3 = self.fusion3(encoder_output3)

        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)

        encoder_output4 = torch.unsqueeze(encoder_output4, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output4 = torch.cat((encoder_output4, returns[idx][10].unsqueeze(4)), dim=4)
        encoder_output4, _ = torch.max(encoder_output4, 4)
        encoder_output4 = self.fusion4(encoder_output4)

        bottleneck = self.bottleneck(encoder_output4)

        bottleneck = torch.unsqueeze(bottleneck, 4)
        for idx in range(len(self.FastSurferCNNs)):
            bottleneck = torch.cat((bottleneck, returns[idx][13].unsqueeze(4)), dim=4)
        bottleneck, _ = torch.max(bottleneck, 4)
        bottleneck = self.fusion5(bottleneck)

        # =====================================================================
        # Step 7: 解码器（上采样 + skip 连接）
        # =====================================================================
        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)

        # =====================================================================
        # Step 8: ★ 解码特征融合 + 分类
        # =====================================================================
        # 将主融合分支与各 backbone 的解码特征拼接后做分类。
        # 这里是最终「多专家融合」的关键位置。
        decoder_outputs = []
        decoder_outputs.append(decoder_output1)
        for ind_return in returns:
            decoder_outputs.append(ind_return[17])  # 各 backbone 的 decoder_output1

        decoder_outputs_fused = self.fusion_layer(torch.cat(decoder_outputs, dim=1))
        logits = self.classifier.forward(decoder_outputs_fused)

        # 同时保留各 backbone 分支的可加 logits 作为辅助输出。
        # 在推理脚本中，这些辅助输出会和主输出一起参与概率累加。
        logits_list = []
        logits_list.append(logits)
        for ind_return in returns:
            logits += ind_return[0]
            logits_list.append(ind_return[0])

        return logits, logits_list


# =========================================================================
# 10-11. v4 / v4_extended — 进一步扩展版本
# =========================================================================
# v4 和 v3 结构相似，但在 forward 中返回更多辅助信息。
# v4_extended 的 logits_list 额外包含一个 logits_sum
#（各 backbone logits 的总和）。
# =========================================================================

class FastSurferCNN_Fuse_Unet_v4(nn.Module):
    """v4：和 v3 结构类似，但 forward 只返回主 logits（不返回 list）"""
    def __init__(self, params):
        super(FastSurferCNN_Fuse_Unet_v4, self).__init__()
        num_channels = params['num_channels']
        params['num_channels'] = params['num_modality'] * 7
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)
        self.fusion1 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion2 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion3 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion4 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion5 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion_layer = nn.Conv2d(params['num_filters'] * (params['num_modality'] + 1), params['num_filters'], params['kernel_c'], params['stride_conv'])
        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.FastSurferCNNs = nn.ModuleList()
        for idx in range(params['num_modality']):
            params['num_channels'] = num_channels
            backbone_model = FastSurferCNN_return_all(params)
            if params['backbone_model'] is not None:
                print('Loading: %s' % params['backbone_model'][idx])
                model_state = torch.load(params['backbone_model'][idx])
                new_state_dict = OrderedDict()
                for k, v in model_state["model_state_dict"].items():
                    if k[:7] == "module.":
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                backbone_model.load_state_dict(new_state_dict)
                for param in backbone_model.parameters():
                    param.requires_grad = False
            else:
                print('No Backbone!!!')
            self.FastSurferCNNs.append(backbone_model)
        print('Initialize Fuse Unet v4 done!')

    def forward(self, x):
        returns = []
        for idx in range(len(self.FastSurferCNNs)):
            fscnn = self.FastSurferCNNs[idx]
            returns.append(fscnn(x[:, ((idx)*7):((idx+1) * 7), :, :]))
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x[:, :, :, :])
        encoder_output1 = torch.unsqueeze(encoder_output1, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output1 = torch.cat((encoder_output1, returns[idx][1].unsqueeze(4)), dim=4)
        encoder_output1, _ = torch.max(encoder_output1, 4)
        encoder_output1 = self.fusion1(encoder_output1)
        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)
        encoder_output2 = torch.unsqueeze(encoder_output2, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output2 = torch.cat((encoder_output2, returns[idx][4].unsqueeze(4)), dim=4)
        encoder_output2, _ = torch.max(encoder_output2, 4)
        encoder_output2 = self.fusion2(encoder_output2)
        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)
        encoder_output3 = torch.unsqueeze(encoder_output3, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output3 = torch.cat((encoder_output3, returns[idx][7].unsqueeze(4)), dim=4)
        encoder_output3, _ = torch.max(encoder_output3, 4)
        encoder_output3 = self.fusion3(encoder_output3)
        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)
        encoder_output4 = torch.unsqueeze(encoder_output4, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output4 = torch.cat((encoder_output4, returns[idx][10].unsqueeze(4)), dim=4)
        encoder_output4, _ = torch.max(encoder_output4, 4)
        encoder_output4 = self.fusion4(encoder_output4)
        bottleneck = self.bottleneck(encoder_output4)
        bottleneck = torch.unsqueeze(bottleneck, 4)
        for idx in range(len(self.FastSurferCNNs)):
            bottleneck = torch.cat((bottleneck, returns[idx][13].unsqueeze(4)), dim=4)
        bottleneck, _ = torch.max(bottleneck, 4)
        bottleneck = self.fusion5(bottleneck)
        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)
        decoder_outputs = []
        decoder_outputs.append(decoder_output1)
        for ind_return in returns:
            decoder_outputs.append(ind_return[17])
        decoder_outputs_fused = self.fusion_layer(torch.cat(decoder_outputs, dim=1))
        logits = self.classifier.forward(decoder_outputs_fused)
        return logits


class FastSurferCNN_Fuse_Unet_v4_extended(nn.Module):
    """v4 extended：logits_list 最后额外包含各 backbone logits 之和"""
    def __init__(self, params):
        super(FastSurferCNN_Fuse_Unet_v4_extended, self).__init__()
        num_channels = params['num_channels']
        params['num_channels'] = params['num_modality'] * 7
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)
        self.fusion1 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion2 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion3 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion4 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion5 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion_layer = nn.Conv2d(params['num_filters'] * (params['num_modality'] + 1), params['num_filters'], params['kernel_c'], params['stride_conv'])
        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.FastSurferCNNs = nn.ModuleList()
        for idx in range(params['num_modality']):
            params['num_channels'] = num_channels
            backbone_model = FastSurferCNN_return_all(params)
            if params['backbone_model'] is not None:
                model_state = torch.load(params['backbone_model'][idx])
                new_state_dict = OrderedDict()
                for k, v in model_state["model_state_dict"].items():
                    if k[:7] == "module.":
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                backbone_model.load_state_dict(new_state_dict)
                for param in backbone_model.parameters():
                    param.requires_grad = False
            else:
                print('No Backbone!!!')
            self.FastSurferCNNs.append(backbone_model)
        print('Initialize Fuse Unet v4 done!')

    def forward(self, x):
        returns = []
        for idx in range(len(self.FastSurferCNNs)):
            fscnn = self.FastSurferCNNs[idx]
            returns.append(fscnn(x[:, ((idx)*7):((idx+1) * 7), :, :]))
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x[:, :, :, :])
        encoder_output1 = torch.unsqueeze(encoder_output1, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output1 = torch.cat((encoder_output1, returns[idx][1].unsqueeze(4)), dim=4)
        encoder_output1, _ = torch.max(encoder_output1, 4)
        encoder_output1 = self.fusion1(encoder_output1)
        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)
        encoder_output2 = torch.unsqueeze(encoder_output2, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output2 = torch.cat((encoder_output2, returns[idx][4].unsqueeze(4)), dim=4)
        encoder_output2, _ = torch.max(encoder_output2, 4)
        encoder_output2 = self.fusion2(encoder_output2)
        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)
        encoder_output3 = torch.unsqueeze(encoder_output3, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output3 = torch.cat((encoder_output3, returns[idx][7].unsqueeze(4)), dim=4)
        encoder_output3, _ = torch.max(encoder_output3, 4)
        encoder_output3 = self.fusion3(encoder_output3)
        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)
        encoder_output4 = torch.unsqueeze(encoder_output4, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output4 = torch.cat((encoder_output4, returns[idx][10].unsqueeze(4)), dim=4)
        encoder_output4, _ = torch.max(encoder_output4, 4)
        encoder_output4 = self.fusion4(encoder_output4)
        bottleneck = self.bottleneck(encoder_output4)
        bottleneck = torch.unsqueeze(bottleneck, 4)
        for idx in range(len(self.FastSurferCNNs)):
            bottleneck = torch.cat((bottleneck, returns[idx][13].unsqueeze(4)), dim=4)
        bottleneck, _ = torch.max(bottleneck, 4)
        bottleneck = self.fusion5(bottleneck)
        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)
        decoder_outputs = []
        decoder_outputs.append(decoder_output1)
        for ind_return in returns:
            decoder_outputs.append(ind_return[17])
        decoder_outputs_fused = self.fusion_layer(torch.cat(decoder_outputs, dim=1))
        logits = self.classifier.forward(decoder_outputs_fused)
        logits_list = []
        logits_list.append(logits)
        logits_sum = logits
        for ind_return in returns:
            logits_list.append(ind_return[0])
            logits_sum += ind_return[0]
        logits_list.append(logits_sum)
        return logits, logits_list

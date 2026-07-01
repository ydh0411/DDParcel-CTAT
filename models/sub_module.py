
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
# sub_module.py — DDParcel / FastSurfer 网络积木块
# =========================================================================
# 本文件定义了构建整个 U-Net 所需的全部「基础模块」：
#
# 1. CompetitiveDenseBlock      — 核心特征提取块（3 层卷积 + maxout 竞争）
# 2. CompetitiveDenseBlockInput — 输入专用的变体（多一层输入 BN）
# 3. CompetitiveEncoderBlock    — 编码块 = DenseBlock + MaxPool（保留索引供上采样）
# 4. CompetitiveEncoderBlockInput — 输入编码块
# 5. CompetitiveDecoderBlock    — 解码块 = Unpool + skip 融合 + DenseBlock
# 6. ClassifierBlock            — 1x1 卷积做逐像素分类
#
# === 核心设计理念：竞争式 maxout（为什么不用 concat？）===
# 传统 U-Net 在 skip-connection 处做通道拼接（concat），导致通道数膨胀。
# 本网络在每次融合点用 maxout（逐元素取最大值）替代 concat：
#   - 两个候选特征图 [C×H×W] → 在额外维度叠加 → 沿该维度取 max
#   - 结果是「每像素只保留更强的那个特征值」
#   - 通道数不增加，计算量更可控，且引入一种「竞争/选优」的归纳偏置
# =========================================================================
import torch
import torch.nn as nn


# =========================================================================
# Block 1: CompetitiveDenseBlock（竞争式密集连接块）
# =========================================================================
# 网络结构中最基本、最常用的特征提取单元。
# 它由 3 个卷积层堆叠而成，每层之间通过「maxout 竞争」来融合输入和输出。
#
# 数据流：
#   输入 x → PReLU → Conv0 ─┬─→ BN → maxout(x, Conv0_out) → PReLU
#                           │
#                    (跳过连接)
#               → Conv1 ─┬─→ BN → maxout(上一步输出, Conv1_out) → PReLU
#                        │
#                 (再次跳过连接)
#              → Conv2(1×1) → BN → 输出
#
# 关键观察：
# - 每层卷积后都和「未经过该层的版本」做 maxout
# - 这本质是一种「残差 + 竞争」思想：模型可以选择「保留原特征」还是「用新特征」
# - 1x1 卷积在最后做通道调整
# =========================================================================
class CompetitiveDenseBlock(nn.Module):
    """
    Function to define a competitive dense block comprising of 3 convolutional layers, with BN/ReLU

    Inputs:
    -- Params
     params = {'num_channels': 1,
               'num_filters': 64,
               'kernel_h': 5,
               'kernel_w': 5,
               'stride_conv': 1,
               'pool': 2,
               'stride_pool': 2,
               'num_classes': 44
               'kernel_c':1
               'input':True
               }
    """

    def __init__(self, params, outblock=False):
        """
        Constructor to initialize the Competitive Dense Block
        :param dict params: dictionary with parameters specifiying block architecture
        :param bool outblock: Flag indicating if last block (before classifier block) is set up.
                               Default: False
        :return None:
        """
        super(CompetitiveDenseBlock, self).__init__()

        # Padding to get output tensor of same dimensions
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        # Sub-layer output sizes for BN; and
        conv0_in_size = int(params['num_filters'])  # num_channels
        conv1_in_size = int(params['num_filters'])
        conv2_in_size = int(params['num_filters'])

        # Define the learnable layers

        # 卷积 0：标准 3×3（或 5×5）卷积，这是第一层特征提取。
        # 注意输入通道是 num_filters（因为前一层已经对齐到了 num_filters）。
        self.conv0 = nn.Conv2d(in_channels=conv0_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        # 卷积 1：和第二层特征提取，结构和卷积 0 相同。
        self.conv1 = nn.Conv2d(in_channels=conv1_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        # 卷积 2：1×1 卷积，用于调整通道数或跨块信息传递。
        # 这里 kernel_c 通常是 1，所以是 1×1 卷积，
        # 作用是在不改变空间维度的前提下融合各通道信息。
        self.conv2 = nn.Conv2d(in_channels=conv2_in_size, out_channels=params['num_filters'],
                               kernel_size=(1, 1),
                               stride=params['stride_conv'], padding=(0, 0))

        # 批归一化层：每层卷积后都接 BN，稳定训练
        self.bn1 = nn.BatchNorm2d(num_features=conv1_in_size)
        self.bn2 = nn.BatchNorm2d(num_features=conv2_in_size)
        self.bn3 = nn.BatchNorm2d(num_features=conv2_in_size)

        # PReLU = Parametric ReLU，带可学习参数的 ReLU
        # 相比普通 ReLU，PReLU 在负数区间有一个可学习的斜率，
        # 给网络多一点点灵活性。
        self.prelu = nn.PReLU()
        # outblock 标志：如果是最后一个块（输出接分类器），
        # 则最后一层不做 BN（因为分类器不需要 BN 后的分布）。
        self.outblock = outblock

    def forward(self, x):
        """
        CompetitiveDenseBlock's computational Graph
        {in (Conv - BN from prev. block) -> PReLU} -> {Conv -> BN -> Maxout -> PReLU} x 2 -> {Conv -> BN} -> out
        end with batch-normed output to allow maxout across skip-connections

        :param tensor x: input tensor (image or feature map)
        :return tensor out: output tensor (processed feature map)
        """
        # =====================================================================
        # 第一步：激活 + 卷积 0
        # =====================================================================
        # Activation from pooled input
        x0 = self.prelu(x)

        # 卷积块 1：与输入特征做竞争式 maxout 融合。
        # maxout 可以理解为「两个候选特征图逐像素取更强响应」。
        x0 = self.conv0(x0)
        x1_bn = self.bn1(x0)
        x0_bn = torch.unsqueeze(x, 4)       # ← 原始输入（跳跃过来的）
        x1_bn = torch.unsqueeze(x1_bn, 4)   # ← 卷积 0 的输出
        x1 = torch.cat((x1_bn, x0_bn), dim=4)  # 沿第 5 维堆叠：NB x C x H x W x 2
        x1_max, _ = torch.max(x1, 4)           # 沿堆叠的维度取 max = 竞争！
        x1 = self.prelu(x1_max)

        # =====================================================================
        # 第二步：卷积 1，再次 maxout
        # =====================================================================
        # 卷积块 2：再次 maxout，仅保留每个位置最强响应。
        # 这种竞争机制替代了常见的 concat + conv 融合方式。
        x1 = self.conv1(x1)
        x2_bn = self.bn2(x1)
        x2_bn = torch.unsqueeze(x2_bn, 4)     # 卷积 1 的输出
        x1_max = torch.unsqueeze(x1_max, 4)   # 上一步 maxout 的结果（作为跳跃连接）
        x2 = torch.cat((x2_bn, x1_max), dim=4)
        x2_max, _ = torch.max(x2, 4)
        x2 = self.prelu(x2_max)

        # =====================================================================
        # 第三步：1×1 卷积（通道融合）—— 输出
        # =====================================================================
        # Convolution block 3 (end with batch-normed output to allow maxout across skip-connections)
        out = self.conv2(x2)

        if not self.outblock:
            out = self.bn3(out)

        return out


# =========================================================================
# Block 2: CompetitiveDenseBlockInput（输入专用版本）
# =========================================================================
# 与 CompetitiveDenseBlock 的唯一区别：
# - 输入先通过 BN（x0_bn = self.bn0(x)），而不是先 PReLU
# - 这是因为输入图像的分布和中间特征图不同，
#   BN 能将原始像素值归一化到合适的范围再进入卷积
#
# 正常块：        x → PReLU → Conv → BN → ...
# 输入块：  x → BN → Conv → BN → PReLU → ...
# =========================================================================
class CompetitiveDenseBlockInput(nn.Module):
    """
    Function to define a competitive dense block comprising of 3 convolutional layers, with BN/ReLU for input

    Inputs:
    -- Params
     params = {'num_channels': 1,
               'num_filters': 64,
               'kernel_h': 5,
               'kernel_w': 5,
               'stride_conv': 1,
               'pool': 2,
               'stride_pool': 2,
               'num_classes': 44
               'kernel_c':1
               'input':True
              }
    """

    def __init__(self, params):
        """
        Constructor to initialize the Competitive Dense Block
        :param dict params: dictionary with parameters specifiying block architecture
        """
        super(CompetitiveDenseBlockInput, self).__init__()

        # Padding to get output tensor of same dimensions
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        # Sub-layer output sizes for BN; and
        # 注意：这里的 conv0_in_size 是 params['num_channels']（原始通道数），
        # 而普通块用的是 params['num_filters']（已经对齐的通道数）。
        # 这是因为输入块是网络的第一层，输入通道数由数据决定（如 7 或 28）。
        conv0_in_size = int(params['num_channels'])
        conv1_in_size = int(params['num_filters'])
        conv2_in_size = int(params['num_filters'])

        # Define the learnable layers
        self.conv0 = nn.Conv2d(in_channels=conv0_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        self.conv1 = nn.Conv2d(in_channels=conv1_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        # 1 \times 1 convolution for the last block
        self.conv2 = nn.Conv2d(in_channels=conv2_in_size, out_channels=params['num_filters'],
                               kernel_size=(1, 1),
                               stride=params['stride_conv'], padding=(0, 0))

        # 多了 bn0（输入 BN），这是与普通块的关键区别
        self.bn0 = nn.BatchNorm2d(num_features=conv0_in_size)
        self.bn1 = nn.BatchNorm2d(num_features=conv1_in_size)
        self.bn2 = nn.BatchNorm2d(num_features=conv2_in_size)
        self.bn3 = nn.BatchNorm2d(num_features=conv2_in_size)

        self.prelu = nn.PReLU()

    def forward(self, x):
        """
        CompetitiveDenseBlockInput's computational Graph
        in -> BN -> {Conv -> BN -> PReLU} -> {Conv -> BN -> Maxout -> PReLU} -> {Conv -> BN} -> out

        :param tensor x: input tensor (image or feature map)
        :return tensor out: output tensor (processed feature map)
        """
        # =====================================================================
        # 输入 BN：把原始像素值的分布拉到 N(0,1) 附近
        # =====================================================================
        # Input batch normalization
        x0_bn = self.bn0(x)

        # Convolution block1
        x0 = self.conv0(x0_bn)
        x1_bn = self.bn1(x0)
        x1 = self.prelu(x1_bn)

        # Convolution block2
        x1 = self.conv1(x1)
        x2_bn = self.bn2(x1)
        # First Maxout
        x1_bn = torch.unsqueeze(x1_bn, 4)
        x2_bn = torch.unsqueeze(x2_bn, 4)  # Add Singleton Dimension along 5th
        x2 = torch.cat((x2_bn, x1_bn), dim=4)  # Concatenating along the 5th dimension
        x2_max, _ = torch.max(x2, 4)
        x2 = self.prelu(x2_max)

        # Convolution block 3
        out = self.conv2(x2)
        out = self.bn3(out)

        return out


# =========================================================================
# Block 3: CompetitiveEncoderBlock（编码块）
# =========================================================================
# 结构：CompetitiveDenseBlock + MaxPool2d
#
# 前向数据流：
#   x → CompetitiveDenseBlock（提取特征）→ MaxPool（下采样 2 倍）→ 输出给下一层
#                           ↓
#                   out_block（跳跃连接）
#                   indices（池化索引，用于解码器反池化）
#
# 返回值有三个：
# - out_encoder:  下采样后的特征图（传给更深层）
# - out_block:    下采样前的特征图（作为 skip-connection）
# - indices:      MaxPool 的索引（解码器用 MaxUnpool 恢复位置）
# =========================================================================
class CompetitiveEncoderBlock(CompetitiveDenseBlock):
    """
    Encoder Block = CompetitiveDenseBlock + Max Pooling
    """

    def __init__(self, params):
        """
        Encoder Block initialization
        :param dict params: parameters like number of channels, stride etc.
        """
        super(CompetitiveEncoderBlock, self).__init__(params)
        # MaxPool2d 的 return_indices=True 非常关键！
        # 这样解码器端的 MaxUnpool2d 就可以利用这些索引，
        # 将小特征图「展开」到原来的位置，保持空间对应关系。
        self.maxpool = nn.MaxPool2d(kernel_size=params['pool'], stride=params['stride_pool'],
                                    return_indices=True)

    def forward(self, x):
        """
        Computational graph for Encoder Block:
          * CompetitiveDenseBlock
          * Max Pooling (+ retain indices)

        :param tensor x: feature map from previous block
        :return: original feature map, maxpooled feature map, maxpool indices
        """
        # 先做特征提取，得到跳跃连接用的特征图
        out_block = super(CompetitiveEncoderBlock, self).forward(x)
        # 再下采样，送给下一层编码器
        out_encoder, indices = self.maxpool(out_block)
        return out_encoder, out_block, indices


# =========================================================================
# Block 4: CompetitiveEncoderBlockInput（输入编码块）
# =========================================================================
# 和上一个块相同，但内部用的是 CompetitiveDenseBlockInput。
# 这是网络的第一级编码器。
# =========================================================================
class CompetitiveEncoderBlockInput(CompetitiveDenseBlockInput):
    """
    Encoder Block = CompetitiveDenseBlockInput + Max Pooling
    """

    def __init__(self, params):
        """
        Encoder Block initialization
        :param dict params: parameters like number of channels, stride etc.
        """
        super(CompetitiveEncoderBlockInput, self).__init__(params)
        self.maxpool = nn.MaxPool2d(kernel_size=params['pool'], stride=params['stride_pool'],
                                    return_indices=True)

    def forward(self, x):
        """
        Computational graph for Encoder Block:
          * CompetitiveDenseBlockInput
          * Max Pooling (+ retain indices)

        :param tensor x: feature map from previous block
        :return: original feature map, maxpooled feature map, maxpool indices
        """
        out_block = super(CompetitiveEncoderBlockInput, self).forward(x)
        out_encoder, indices = self.maxpool(out_block)
        return out_encoder, out_block, indices


# =========================================================================
# Block 5: CompetitiveDecoderBlock（解码块）
# =========================================================================
# 结构：MaxUnpool → maxout 竞争融合 → CompetitiveDenseBlock
#
# 解码器做的事情：
# 1. 将深层的特征图通过 MaxUnpool 上采样 2 倍
#    （使用编码器保存的池化索引，保证位置对齐）
# 2. 将上采样结果和编码器对应层的跳跃连接特征做 maxout 竞争
# 3. 送入 CompetitiveDenseBlock 进一步处理
#
# 为什么用 Unpool 而不是转置卷积？
# - 转置卷积有可学习参数，但可能产生棋盘伪影
# - MaxUnpool 没有参数，配合编码器的索引，能精准恢复空间结构
# - 配合 maxout 竞争，效果接近但更轻量
# =========================================================================
class CompetitiveDecoderBlock(CompetitiveDenseBlock):
    """
    Decoder Block = (Unpooling + Skip Connection) --> Dense Block
    """

    def __init__(self, params, outblock=False):
        """
        Decoder Block initialization
        :param dict params: parameters like number of channels, stride etc.
        :param bool outblock: Flag, indicating if last block of network before classifier
                              is created. Default: False
        """
        super(CompetitiveDecoderBlock, self).__init__(params, outblock=outblock)
        # MaxUnpool2d：将小特征图按索引展开为大特征图
        # kernel_size 和 stride 必须和编码器的 MaxPool 一致
        self.unpool = nn.MaxUnpool2d(kernel_size=params['pool'], stride=params['stride_pool'])

    def forward(self, x, out_block, indices):
        """
        Computational graph Decoder block:
          * Unpooling of feature maps from lower block
          * Maxout combination of unpooled map + skip connection
          * Forwarding toward CompetitiveDenseBlock

        :param tensor x: input feature map from lower block (gets unpooled and maxed with out_block)
        :param tensor out_block: skip connection feature map from the corresponding Encoder
        :param tensor indices: indices for unpooling from the corresponding Encoder (maxpool op)
        :return: processed feature maps
        """
        # Unpool 复用编码器保存的池化索引，以保持空间对应关系。
        # 这比简单插值更「对齐」编码器阶段的池化位置。
        unpool = self.unpool(x, indices)
        unpool = torch.unsqueeze(unpool, 4)

        out_block = torch.unsqueeze(out_block, 4)
        # 竞争式卷积融合（maxout）而非拼接（concat）
        concat = torch.cat((unpool, out_block), dim=4)
        concat_max, _ = torch.max(concat, 4)
        out_block = super(CompetitiveDecoderBlock, self).forward(concat_max)

        return out_block


# =========================================================================
# Block 6: ClassifierBlock（分类头）
# =========================================================================
# 网络最后一层：一个 1×1 卷积，将特征图映射到类别数。
#
# 例如：输入是 64 通道的特征图，num_classes=82，
# 则输出是 82 通道的特征图，每个通道对应一个类别的 logits。
# 之后在推理时对通道维度取 argmax，就得到每个像素的类别标签。
#
# 注意：这里没有 softmax！因为 CrossEntropyLoss 内部会做 softmax。
# =========================================================================
class ClassifierBlock(nn.Module):
    """
    Classification Block
    """
    def __init__(self, params):
        """
        Classifier Block initialization
        :param dict params: parameters like number of channels, stride etc.
        """
        super(ClassifierBlock, self).__init__()
        # 1x1 卷积：将 num_channels → num_classes
        # kernel_c 通常为 1，stride_conv 通常为 1
        self.conv = nn.Conv2d(params['num_channels'], params['num_classes'], params['kernel_c'],
                              params['stride_conv'])

    def forward(self, x):
        """
        Computational graph of classifier
        :param tensor x: output of last CompetitiveDenseDecoder Block-
        :return: logits
        """
        logits = self.conv(x)

        return logits

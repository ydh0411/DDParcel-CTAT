
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
# losses.py — 训练阶段使用的损失函数
# =========================================================================
# 注意：推理（DDSurfer_Pred.py）不需要本文件的任何类。
# 这些损失仅在训练时用于优化网络参数。
#
# 本文件定义了三种损失：
# 1. DiceLoss              — 基于 Dice 系数的损失（衡量分割区域重叠度）
# 2. CrossEntropy2D        — 逐像素交叉熵（衡量逐类别的预测置信度）
# 3. CombinedLoss          — 上述两者的加权和（DDParcel 训练用的主损失）
# 4. CombinedLoss_fuse_Unet_v2_extended — 融合网络多路输出的组合损失
#
# === 为什么同时用 Dice 和 CE？ ===
# - Dice Loss：直接优化分割精度，对类别不平衡不敏感
#   （小区域和大区域的 Dice 权重相似）
# - CE Loss：逐像素独立优化，收敛更快但易被大区域主导
# - 两者加权结合 = 兼顾「轮廓准确性」和「逐像素置信度」
# =========================================================================
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


# =========================================================================
# DiceLoss — 基于 Dice 系数的损失函数
# =========================================================================
# Dice 系数是医学图像分割中最常用的评价指标之一：
#   Dice = 2 * |P ∩ T| / (|P| + |T|)
#   其中 P = 预测区域，T = 真实区域
#
# Dice Loss = 1 - Dice（越小越好）
#
# 特点：
# - 对类别不平衡鲁棒（小结构和大结构的权重相当）
# - 直接优化分割交叠度
# - 需要先将预测转为 one-hot 编码
# =========================================================================
class DiceLoss(_Loss):
    """
    Dice Loss
    """

    def forward(self, output, target, weights=None, ignore_index=None):
        """
        :param output: N x C x H x W Variable（softmax 后的概率图）
        :param target: N x H x W LongTensor，每个像素是类别索引（从 0 开始）
        :param weights: C FloatTensor，每个类别的权重
        :param int ignore_index: 忽略索引 x，不参与损失计算（如边界填充区域）
        :return: 标量 loss 值
        """
        eps = 0.001  # 防止除零

        # 将 target（类别索引）转为 one-hot 编码（和 output 形状一致）
        encoded_target = output.detach() * 0

        if ignore_index is not None:
            # 如果有需要忽略的像素（如图像边界填充部分），将其设置为 0
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            # scatter_ 将 target 中值为 k 的位置，在 encoded_target 第 k 通道置 1
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        # Dice 分子：2 * 预测和真值的交集
        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)

        # Dice 分母：预测 + 真值面积之和
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0

        denominator = denominator.sum(0).sum(1).sum(1) + eps
        # 每个类别的 Dice Loss
        loss_per_channel = weights * (1 - (numerator / denominator))

        # 对所有类别的 loss 取平均
        return loss_per_channel.sum() / output.size(1)


# =========================================================================
# CrossEntropy2D — 2D 交叉熵损失
# =========================================================================
# 标准的多分类交叉熵损失。PyTorch 的 CrossEntropyLoss 内部做了
# LogSoftmax + NLLLoss，所以输入 raw logits 即可。
#
# 常用于语义分割中评估逐像素的分类准确性。
# 和 DiceLoss 互补：CE 更关注逐点的一致性，Dice 更关注整体区域。
# =========================================================================
class CrossEntropy2D(nn.Module):
    """
    2D Cross-entropy loss implemented as negative log likelihood
    """

    def __init__(self, weight=None, reduction='none'):
        super(CrossEntropy2D, self).__init__()
        # reduction='none' 表示返回逐像素 loss，
        # 后续会和权重图逐元素相乘再求均值
        self.nll_loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)


# =========================================================================
# CombinedLoss — Dice + CE 组合损失
# =========================================================================
# DDParcel 训练阶段实际使用的损失函数。
# 它结合了 Dice Loss（区域重叠度）和 CrossEntropy Loss（逐像素精度）。
#
# loss = w_dice * DiceLoss + w_ce * CrossEntropyLoss
#
# 训练时需要对权重图加权（weighted CE），
# 权重图由 create_weight_mask() 生成（中位数频率平衡 + 边缘加权）。
# =========================================================================
class CombinedLoss(nn.Module):
    """
    For CrossEntropy the input has to be a long tensor
    Args:
        -- inputx N x C x H x W
        -- target - N x H x W - int type
        -- weight - N x H x W - float（逐像素权重图）
    """

    def __init__(self, weight_dice=1, weight_ce=1):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropy2D()
        self.dice_loss = DiceLoss()
        self.weight_dice = weight_dice   # Dice 项权重（默认 1）
        self.weight_ce = weight_ce       # CE 项权重（默认 1）

    def forward(self, inputx, target, weight):
        target = target.type(torch.LongTensor)  # CE 要求 target 为 Long 类型
        if inputx.is_cuda:
            target = target.cuda()

        # 对 logits 做 softmax 得到概率分布（Dice 需要概率输入）
        input_soft = F.softmax(inputx, dim=1)
        dice_val = torch.mean(self.dice_loss(input_soft, target))
        # CE 接收 raw logits，乘以权重图（突出边缘和稀有类别）
        ce_val = torch.mean(torch.mul(self.cross_entropy_loss.forward(inputx, target), weight))

        # 加权组合
        total_loss = torch.add(torch.mul(dice_val, self.weight_dice), torch.mul(ce_val, self.weight_ce))

        return total_loss, dice_val, ce_val


# =========================================================================
# CombinedLoss_fuse_Unet_v2_extended — 融合网络的多路组合损失
# =========================================================================
# 对于多模态融合网络（如 v2_extended），每个 backbone 也有自己的 logits。
# 这个损失函数对「主融合输出 + 各 backbone 独立输出」都计算损失求和。
#
# 这样可以确保每个 backbone 自身也能做出合理的预测，
# 而不仅仅是融合分支在学习。
# =========================================================================
class CombinedLoss_fuse_Unet_v2_extended(nn.Module):
    """
    For CrossEntropy the input has to be a long tensor
    Args:
        -- inputxs: list of (N x C x H x W) tensors [主输出, backbone0, backbone1, ...]
        -- target - N x H x W - int type
        -- weight - N x H x W - float
    """

    def __init__(self, weight_dice=1, weight_ce=1):
        super(CombinedLoss_fuse_Unet_v2_extended, self).__init__()
        self.cross_entropy_loss = CrossEntropy2D()
        self.dice_loss = DiceLoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, inputxs, target, weight):
        target = target.type(torch.LongTensor)

        # ★ 遍历所有输出路径（主融合 + 各 backbone）分别计算损失
        for inputx in inputxs:
            if inputx.is_cuda:
                target = target.cuda()

            input_soft = F.softmax(inputx, dim=1)
            dice_val = torch.mean(self.dice_loss(input_soft, target))
            ce_val = torch.mean(torch.mul(self.cross_entropy_loss.forward(inputx, target), weight))
            total_loss = torch.add(torch.mul(dice_val, self.weight_dice), torch.mul(ce_val, self.weight_ce))

        # 对所有输出路径的损失求和
        total_loss = torch.sum(total_loss)
        dice_val = torch.sum(dice_val)
        ce_val = torch.sum(ce_val)

        return total_loss, dice_val, ce_val

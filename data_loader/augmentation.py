
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
# augmentation.py — 数据预处理与增强
# =========================================================================
# 本文件提供了训练和推理阶段共用的数据变换操作。
#
# 两类变换：
# 1. 推理路径（ToTensorTest）：
#    - NumPy → Tensor
#    - 像素值归一化到 [0, 1]
#    - H×W×C → C×H×W（PyTorch 格式）
#
# 2. 训练路径：
#    - ToTensor：类似 ToTensorTest，但处理 dict 格式样本
#    - AugmentationPadImage：填充图像边界（反射填充或补零）
#    - AugmentationRandomCrop：随机/中心裁剪
#    - ExtractSlice / ExtractSliceTest：从厚切片中提取特定切片
#
# === 厚切片（Thick Slice）核心概念 ===
# DDParcel 网络不是直接处理 3D 体数据，而是将 3D 体数据
# 转换为 2.5D 的「厚切片」表示：
# - 每个目标切片的前后各取 n 个相邻切片一起作为输入
# - n=3 时共 7 个切片
# - 这些切片沿通道维堆叠，形成 C×H×W 的 2D 输入
# - 网络只需分割中间那张切片
# - 这样在 2D 网络中利用了 3D 上下文信息
# =========================================================================
import numpy as np
import torch


# =========================================================================
# 推理变换
# =========================================================================

class ToTensorTest(object):
    """
    推理用的数据变换：将 numpy 数组转为 PyTorch Tensor。

    做的事情很简单：
    1. 转为 float32
    2. 除以 255 归一化到 [0, 1]
    3. 从 H×W×C 转置为 C×H×W（PyTorch 卷积需要的格式）

    为什么除以 255？
    - 因为 conform 阶段将图像像素缩放到 [0, 255] 的 uint8 范围
    - 除以 255 回到 [0, 1] 区间，适应网络期望的输入分布
    """

    def __call__(self, img):
        img = img.astype(np.float32)

        # Normalize and clamp between 0 and 1
        img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))

        return img


# =========================================================================
# 训练变换
# =========================================================================

class ToTensor(object):
    """
    训练用的数据变换（和 ToTensorTest 类似，但处理 dict 格式）。

    输入是一个包含 'img', 'label', 'weight' 的 dict，
    只对 img 做归一化和转置，label 和 weight 保持原样（不除以 255）。
    """

    def __call__(self, sample):
        img, label, weight = sample['img'], sample['label'], sample['weight']

        img = img.astype(np.float32)

        # Normalize image and clamp between 0 and 1
        img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))

        return {'img': torch.from_numpy(img), 'label': label, 'weight': weight}


class AugmentationPadImage(object):
    """
    图像填充增强。

    在训练前对图像做边界填充，使后续随机裁剪可以在更多位置上采样。
    填充方式默认为 'edge'（反射填充/边缘复制），
    这意味着填充区域的值取自图像边缘，而不是补零。

    为什么需要填充？
    - 随机裁剪需要图像大于裁剪尺寸
    - 如果原图刚好等于裁剪尺寸，padding 可以提供采样空间
    """

    def __init__(self, pad_size=((16, 16), (16, 16)), pad_type="edge"):

        assert isinstance(pad_size, (int, tuple))

        if isinstance(pad_size, int):
            # 不对通道维做 padding
            self.pad_size_image = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
            self.pad_size_mask = ((pad_size, pad_size), (pad_size, pad_size))

        else:
            self.pad_size = pad_size

        self.pad_type = pad_type

    def __call__(self, sample):
        img, label, weight = sample['img'], sample['label'], sample['weight']

        img = np.pad(img, self.pad_size_image, self.pad_type)
        label = np.pad(label, self.pad_size_mask, self.pad_type)
        weight = np.pad(weight, self.pad_size_mask, self.pad_type)

        return {'img': img, 'label': label, 'weight': weight}


class AugmentationRandomCrop(object):
    """
    随机/中心裁剪增强。

    训练时使用 Random 模式：
    - 随机的左上角起始位置，每次看到图像的不同区域
    - 相当于免费的数据增强，增加模型对空间偏移的鲁棒性

    推理时使用 Center 模式：
    - 取图像中心区域，保证结果可复现
    """

    def __init__(self, output_size, crop_type='Random'):

        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

        self.crop_type = crop_type

    def __call__(self, sample):
        img, label, weight = sample['img'], sample['label'], sample['weight']

        h, w, _ = img.shape

        if self.crop_type == 'Center':
            top = (h - self.output_size[0]) // 2
            left = (w - self.output_size[1]) // 2

        else:
            top = np.random.randint(0, h - self.output_size[0])
            left = np.random.randint(0, w - self.output_size[1])

        bottom = top + self.output_size[0]
        right = left + self.output_size[1]

        img = img[top:bottom, left:right, :]
        label = label[top:bottom, left:right]
        weight = weight[top:bottom, left:right]

        return {'img': img, 'label': label, 'weight': weight}


class ExtractSlice(object):
    """
    从厚切片张量中提取特定切片（训练）。

    在厚切片构造中，每个目标位置对应 7 个相邻切片堆叠。
    sliceID 指定取 7 个中的哪一个：
    - sliceID=0: 最靠前的切片
    - sliceID=3: 中间切片（目标切片）
    - sliceID=6: 最靠后的切片

    训练时可以使用不同 sliceID 来做数据增强。
    """

    def __init__(self, sliceID):
        self.sliceID = sliceID

    def __call__(self, sample):
        img, label, weight = sample['img'], sample['label'], sample['weight']
        # 厚切片通道是 7 的倍数，每 7 个一组代表一个目标切片的全部邻域
        img = img[:, :, self.sliceID::7]

        return {'img': img, 'label': label, 'weight': weight}


class ExtractSliceTest(object):
    """
    从厚切片张量中提取特定切片（推理）。

    和 ExtractSlice 功能相同，但输入是单个 numpy 数组而不是 dict。
    """

    def __init__(self, sliceID):
        self.sliceID = sliceID

    def __call__(self, img):
        img = img[:, :, self.sliceID::7]

        return img


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
# load_neuroimaging_data.py — 数据加载与预处理核心模块
# =========================================================================
# 本文件是 DDParcel 项目中最复杂的数据处理模块，包含：
#
# 1. 数据加载与空间一致性化
#    - load_and_conform_image: 读取 MRI 并统一空间
#
# 2. 视角坐标变换
#    - transform_axial / transform_sagittal: 体数据轴置换
#
# 3. 厚切片构造
#    - get_thick_slices: 3D → 2.5D 厚切片（核心！）
#    - filter_blank_slices: 过滤空白切片
#
# 4. 权重图生成
#    - create_weight_mask: 中位数频率平衡 + 边缘加权
#
# 5. 标签后处理
#    - fill_unknown_labels_per_hemi: 填充皮层未知区域
#    - fill_WMhyper_per_hemi: 处理白质高信号
#
# 6. 标签映射（训练和推理间转换）
#    - map_label2aparc_aseg:     推理：内部索引 → FreeSurfer 标签
#    - map_aparc_aseg2label:     训练：FreeSurfer 标签 → 内部索引
#    - map_prediction_sagittal2full: 矢状面特殊标签映射
#
# 7. 连通域分析
#    - bbox_3d: 3D 包围盒
#    - get_largest_cc: 最大连通域
#
# 8. PyTorch Dataset 类
#    - OrigDataThickSlices:                  推理数据集（单模态）
#    - OrigDataThickSlices_Fused_Input:      推理数据集（多模态融合）- ★ 使用
#    - AsegDatasetWithAugmentation:          训练数据集
#    - AsegDatasetWithAugmentation_Fused_Input: 训练数据集（多模态融合）
# =========================================================================
import nibabel as nib
import numpy as np
import h5py
import scipy.ndimage.morphology as morphology
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import sys
import glob, os

from skimage.measure import label
from torch.utils.data.dataset import Dataset
from .conform import is_conform, conform, check_affine_in_nifti


# =========================================================================
# 1. 数据加载与空间一致性化
# =========================================================================

def load_and_conform_image(img_filename, interpol=1, logger=None, imagetype='image'):
    """
    读取 MRI 影像，并在必要时进行 conform（统一到标准空间）。

    为什么需要 conform？
    - 网络训练时默认输入是统一几何空间（256³ + 固定方向）；
    - 临床/科研数据来源多样，直接输入可能方向/spacing 不一致；
    - 先统一空间可以减少模型「看错位置」的风险。

    conform 做了什么：
    1. 重采样到 256×256×256 的网格
    2. 体素尺寸统一为 1mm 各向同性
    3. 方向统一为 RAS（Left → Right, Anterior → Posterior, Superior → Inferior）
    （实际上是 LIA 方向，因为 FreeSurfer 的约定不同）

    :param str img_filename: path and name of volume to read
    :param int interpol: interpolation order for image conformation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :param logger: logger object for logging messages
    :param imagetype: 'image' or 'label'（标签使用 nearest 插值）
    :return: (header_info, affine_info, orig_data)
    """
    orig = nib.load(img_filename)

    if not is_conform(orig):
        if logger is not None:
            logger.info('Conforming image to UCHAR, RAS orientation, and 1mm isotropic voxels')
        else:
            print('Conforming image to RAS orientation, and 1mm isotropic voxels')

        if len(orig.shape) > 3 and orig.shape[3] != 1:
            sys.exit('ERROR: Multiple input frames (' + format(orig.shape[3]) + ') not supported!')

        # Check affine if image is nifti image
        if img_filename[-7:] == ".nii.gz" or img_filename[-4:] == ".nii":
            if not check_affine_in_nifti(orig, logger=logger):
                sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")

        # conform
        orig = conform(orig, interpol, imagetype=imagetype)

    # Collect header and affine information
    header_info = orig.header
    affine_info = orig.affine
    orig = np.asarray(orig.get_fdata(), dtype=np.float32)

    return header_info, affine_info, orig


# =========================================================================
# 2. 视角坐标变换
# =========================================================================
# DDParcel 使用 2.5D 三视角策略：
# - 轴向（Axial）: 从上往下看，切片平面 = XY
# - 冠状（Coronal）: 从前往后看，切片平面 = XZ
# - 矢状（Sagittal）: 从左往右看，切片平面 = YZ
#
# 每个视角都独立训练一个网络，推理时三个视角的结果加权融合。
# 在送入网络前，需要把体数据转到该视角的切片方向。
#
# 这里做的不是插值重采样，而是简单的轴置换（np.moveaxis），
# 所以没有信息损失，速度也快。
# =========================================================================

def transform_axial(vol, coronal2axial=True):
    """
    Function to transform volume into Axial axis and back.

    冠状面 → 轴状面：将轴顺序从 (x, y, z) 变为 (y, z, x)
    恢复到冠状面：反向操作

    直观理解：把体数据「转一下」，使得切片方向变成轴向切片。
    """
    if coronal2axial:
        return np.moveaxis(vol, [0, 1, 2], [1, 2, 0])
    else:
        return np.moveaxis(vol, [0, 1, 2], [2, 0, 1])


def transform_sagittal(vol, coronal2sagittal=True):
    """
    Function to transform volume into Sagittal axis and back.

    冠状面 → 矢状面：将轴顺序从 (x, y, z) 变为 (z, y, x)

    注意：正向和反向变换是一样的（都是 swap x ↔ z），
    因为应用到两次就回到原样了。
    """
    if coronal2sagittal:
        return np.moveaxis(vol, [0, 1, 2], [2, 1, 0])
    else:
        return np.moveaxis(vol, [0, 1, 2], [2, 1, 0])


# =========================================================================
# 3. 厚切片构造 ★ DDParcel 核心概念 ★
# =========================================================================
# 为什么需要厚切片（2.5D）？
#
# 纯 2D 分割：每张切片独立处理 → 缺少沿切片方向的上下文信息，分割不连贯
# 纯 3D 分割：计算量太大，GPU 显存不够
# 2.5D 折中：每个目标切片 + 前后邻域切片一起输入 →
#           在 2D 网络中加入了 3D 上下文 → 效果好且计算可接受
#
# 具体做法：
# 对于 256×256×256 的体数据，对每个切片位置 z：
# 取 [z-3, z-2, z-1, z, z+1, z+2, z+3] 共 7 个切片
# 堆叠成一个 7 通道的 2D 图像（256×256×7）
# 送入 2D 卷积网络，网络只需分割中间的切片 z
# =========================================================================

def get_thick_slices(img_data, slice_thickness=3):
    """
    从 3D 体数据构造「厚切片（2.5D）」输入。

    直观理解：
    - 目标切片的前后各取 `slice_thickness` 层；
    - 中间层是当前要分割的层；
    - 所有层沿通道维堆叠后送入 2D 卷积网络。
    - 当 slice_thickness=3 时，每个模态会产生 7 个通道。

    :param np.ndarray img_data: 3D MRI image read in with nibabel
    :param int slice_thickness: number of slices to stack on top and below slice of interest (default=3)
    :return: H×W×D×(2*slice_thickness+1) 的厚切片张量
    """
    h, w, d = img_data.shape
    # 用边缘值填充首尾，确保边界处的切片也有足够的邻居
    img_data_pad = np.expand_dims(np.pad(img_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode='edge'),
                                  axis=3)
    # 初始化空张量（沿通道维拼接）
    img_data_thick = np.ndarray((h, w, d, 0), dtype=np.uint8)

    # 以每个中心切片为基准，将相邻切片堆叠到通道维。
    for slice_idx in range(2 * slice_thickness + 1):
        img_data_thick = np.append(img_data_thick, img_data_pad[:, :, slice_idx:d + slice_idx, :], axis=3)

    return img_data_thick


def filter_blank_slices_thick(img_vol, label_vol, weight_vol, threshold=50):
    """
    过滤空白切片（标签中像素少于 threshold 的切片被丢弃）。

    训练时使用：很多切片（尤其是脑的顶部和底部）只有很少的脑组织，
    让网络在这些切片上训练意义不大，还拖慢训练速度。
    所以只保留标签像素 > threshold 的切片。
    """
    select_slices = (np.sum(label_vol, axis=(0, 1)) > threshold)

    img_vol = img_vol[:, :, select_slices, :]
    label_vol = label_vol[:, :, select_slices]
    weight_vol = weight_vol[:, :, select_slices]

    return img_vol, label_vol, weight_vol


# =========================================================================
# 4. 权重图生成
# =========================================================================

def create_weight_mask(mapped_aseg, max_weight=5, max_edge_weight=5):
    """
    创建训练用的权重图，包含两部分：

    1. 中位数频率平衡（Median Frequency Balancing）：
       - 统计每类像素数，稀有类别获得更高权重
       - 权重 = 中位数(各类像素数) / 该类像素数
       - 限制最大权重 ≤ max_weight，避免某些极稀有类别主导损失

    2. 边缘加权（Gradient Weighting）：
       - 在分割边界处的像素获得额外权重
       - 这鼓励网络更关注边界区域的精确分割
       - 用分割标签的梯度检测边界位置

    :param mapped_aseg: 分割标签图 [H, W, D]
    :param max_weight: 最大权重上限（默认 5）
    :param max_edge_weight: 边界处的额外权重（默认 5）
    :return: weights_mask [H, W, D]
    """
    unique, counts = np.unique(mapped_aseg, return_counts=True)

    # Median Frequency Balancing
    class_wise_weights = np.median(counts) / counts
    class_wise_weights[class_wise_weights > max_weight] = max_weight
    (h, w, d) = mapped_aseg.shape

    weights_mask = np.reshape(class_wise_weights[mapped_aseg.ravel()], (h, w, d))

    # Gradient Weighting
    (gx, gy, gz) = np.gradient(mapped_aseg)
    grad_weight = max_edge_weight * np.asarray(np.power(np.power(gx, 2) + np.power(gy, 2) + np.power(gz, 2), 0.5) > 0,
                                               dtype=np.float32)

    weights_mask += grad_weight

    return weights_mask


# =========================================================================
# 5. 标签后处理
# =========================================================================

def fill_unknown_labels_per_hemi(gt, unknown_label, cortex_stop):
    """
    填充半球内的未知标签区域。

    在 FreeSurfer 标签空间中，有些区域被标记为「未知皮层」
    （lh: 1000, rh: 2000）。这些区域需要被最近的有效皮层标签替换。

    做法：
    1. 找到未知区域
    2. 膨胀未知区域，找到和它接壤的「已知」皮层标签
    3. 对每个已知标签做高斯模糊扩散
    4. 未知区域的每个体素取扩散值最大的标签（即最近的已知标签）
    """
    h, w, d = gt.shape
    struct1 = ndimage.generate_binary_structure(3, 2)

    unknown = gt == unknown_label
    # 膨胀后取异或 → 未知区域周围一圈的已知标签
    unknown = (morphology.binary_dilation(unknown, struct1) ^ unknown)
    list_parcels = np.unique(gt[unknown])

    # 只保留该半球内的皮层标签
    mask = (list_parcels > unknown_label) & (list_parcels < cortex_stop)
    list_parcels = list_parcels[mask]

    # 高斯扩散每个候选标签
    blur_vals = np.ndarray((h, w, d, 0), dtype=np.float32)

    for idx in range(len(list_parcels)):
        aseg_blur = filters.gaussian_filter(1000 * np.asarray(gt == list_parcels[idx], dtype=np.float32), sigma=5)
        blur_vals = np.append(blur_vals, np.expand_dims(aseg_blur, axis=3), axis=3)

    # 取扩散值最大的标签作为填充
    unknown = np.argmax(blur_vals, axis=3)
    unknown = np.reshape(list_parcels[unknown.ravel()], (h, w, d))

    mask = gt == unknown_label
    gt[mask] = unknown[mask]

    return gt


def fill_WMhyper_per_hemi(gt, WMhyper_label=77, replace_labels=[2, 41]):
    """
    处理白质高信号（WM hyperintensity）标签。

    和 fill_unknown_labels_per_hemi 类似，
    将标签 77（白质高信号）替换为最近的 WM 标签（2 或 41）。
    如果该区域附近都是白质，就用白质填充；
    如果附近有皮层，就保留皮层标签。
    """
    h, w, d = gt.shape
    struct1 = ndimage.generate_binary_structure(3, 2)

    unknown = gt == WMhyper_label
    if np.sum(unknown) == 0:
        return gt

    unknown = (morphology.binary_dilation(unknown, struct1) ^ unknown)
    list_parcels = np.unique(gt[unknown])

    if np.intersect1d(list_parcels, replace_labels).shape[0] > 0:
        list_parcels = np.intersect1d(list_parcels, replace_labels)

    blur_vals = np.ndarray((h, w, d, 0), dtype=np.float32)

    for idx in range(len(list_parcels)):
        aseg_blur = filters.gaussian_filter(1000 * np.asarray(gt == list_parcels[idx], dtype=np.float32), sigma=5)
        blur_vals = np.append(blur_vals, np.expand_dims(aseg_blur, axis=3), axis=3)

    unknown = np.argmax(blur_vals, axis=3)
    unknown = np.reshape(list_parcels[unknown.ravel()], (h, w, d))

    mask = gt == WMhyper_label
    gt[mask] = unknown[mask]

    return gt


# =========================================================================
# 6. 标签映射 — 查找表（LUT）
# =========================================================================
# == 背景知识：FreeSurfer 标签空间 ==
# FreeSurfer 使用特定数字表示每个脑区（如 2=左脑白质, 41=右脑白质,
# 1001=左脑上颞回皮层, 2001=右脑上颞回皮层...）。
# 完整列表见：https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
#
# == 为什么需要标签映射？==
# 网络的输出是 0~N 的内部类别索引（紧凑型），
# FreeSurfer 的标签是分散且稀疏的（如 2, 41, 1001, 2001...）。
# 所以需要 LUT 在两个空间之间转换。
# =========================================================================

def map_label2aparc_aseg(mapped_aseg):
    """
    推理阶段：从网络的内部类别索引 → FreeSurfer aparc+aseg 标签。

    网络输出的是 0~81（或 0~53 对矢状面）的紧凑整数索引，
    这个函数通过 LUT 映射到标准的 FreeSurfer 标签 ID。

    例如：内部索引 0 → 标签 0 (背景)
          内部索引 1 → 标签 2 (左脑白质)
          内部索引 2 → 标签 4 (左脑侧脑室)
          ...

    :param np.ndarray mapped_aseg: 内部标签空间的分割 [H, W, D]
    :return: FreeSurfer 标签空间的分割
    """
    aseg = np.zeros_like(mapped_aseg)
    # 内部类别索引 → FreeSurfer aparc+aseg 标签 ID 的查找表。
    labels = np.array([0, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14,
                       15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44,
                       46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
                       192, 1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011,
                       1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022,
                       1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035,
                       2002, 2005, 2010, 2012, 2013, 2014, 2016, 2017, 2021, 2022, 2023,
                       2024, 2025, 2028])

    h, w, d = aseg.shape

    # 核心操作：用列表作为索引做 LUT 映射
    # labels[mapped_aseg] 等价于：对每个体素，
    # 取 mapped_aseg 中该位置的值作为索引，在 labels 中查找
    aseg = labels[mapped_aseg.ravel()]

    aseg = aseg.reshape((h, w, d))

    return aseg


def map_aparc_aseg2label(aseg, aseg_nocc=None):
    """
    训练阶段：FreeSurfer aparc+aseg 标签 → 内部紧凑类别索引。

    同时返回两种标签映射：
    - mapped_aseg: 用于轴状/冠状网络（82 类）
    - mapped_aseg_sag: 用于矢状网络（54 类）

    主要操作：
    1. 合并左右标签（如 2→41 将左WM合并到右WM）
    2. 合并小结构（如肼胝体 251~255 → 192）
    3. 去除不需要的标签（如视交叉 85 → 0 背景）
    4. 填充未知皮层区域
    5. LUT 映射到紧凑索引
    """
    aseg = aseg.astype(np.int16)

    aseg_temp = aseg.copy()
    # 以下映射将 FreeSurfer 标签统一到更紧凑的表示
    aseg[aseg == 80] = 77   # 低信号区
    aseg[aseg == 85] = 0    # 视交叉→背景
    aseg[aseg == 62] = 41   # 右脑血管→右WM
    aseg[aseg == 30] = 2    # 左脑血管→左WM
    aseg[aseg == 72] = 24   # 第五脑室→CSF

    # 将 WM 标签映射到左右 WM
    aseg[(aseg_temp >= 3000) & (aseg_temp < 3999)] = 2
    aseg[(aseg_temp >= 4000) & (aseg_temp < 4999)] = 41

    aseg[aseg_temp == 5001] = 2
    aseg[aseg_temp == 5002] = 41

    # 肼胝体合并
    aseg[(aseg >= 251) & (aseg <= 255)] = 251

    # 如果有无肼胝体版本，用其替换
    if aseg_nocc is not None:
        cc_mask = (aseg >= 251) & (aseg <= 255)
        aseg[cc_mask] = aseg_nocc[cc_mask]

    aseg[aseg == 3] = 0   # 剩余皮层标签→背景
    aseg[aseg == 42] = 0

    # 填充未知皮层
    if np.any(np.in1d([1000, 2000], aseg.ravel())):
        aseg = fill_unknown_labels_per_hemi(aseg, 1000, 2000)
        aseg = fill_unknown_labels_per_hemi(aseg, 2000, 3000)

    # 将右半球皮层标签映射到左半球
    cortical_label_mask = (aseg >= 2000) & (aseg <= 2999)
    aseg[cortical_label_mask] = aseg[cortical_label_mask] - 1000

    # 保留跨半球接触的皮层标签（不合并到左半球）
    aseg[aseg_temp == 2014] = 2014
    aseg[aseg_temp == 2028] = 2028
    aseg[aseg_temp == 2012] = 2012
    aseg[aseg_temp == 2016] = 2016
    aseg[aseg_temp == 2002] = 2002
    aseg[aseg_temp == 2023] = 2023
    aseg[aseg_temp == 2017] = 2017
    aseg[aseg_temp == 2024] = 2024
    aseg[aseg_temp == 2010] = 2010
    aseg[aseg_temp == 2013] = 2013
    aseg[aseg_temp == 2025] = 2025
    aseg[aseg_temp == 2022] = 2022
    aseg[aseg_temp == 2021] = 2021
    aseg[aseg_temp == 2005] = 2005

    # 轴状/冠状网络的 LUT（82 类）
    labels = np.array([0, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14,
                       15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44,
                       46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
                       77, 251, 1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011,
                       1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022,
                       1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035,
                       2002, 2005, 2010, 2012, 2013, 2014, 2016, 2017, 2021, 2022, 2023,
                       2024, 2025, 2028])

    h, w, d = aseg.shape
    lut_aseg = np.zeros(max(labels) + 1, dtype='int')
    for idx, value in enumerate(labels):
        lut_aseg[value] = idx

    mapped_aseg = lut_aseg.ravel()[aseg.ravel()]
    mapped_aseg = mapped_aseg.reshape((h, w, d))

    # 矢状网络的标签映射（54 类）
    # 矢状网络使用更少的类别，因为左右半球对称
    aseg[aseg == 2] = 41
    aseg[aseg == 3] = 42
    aseg[aseg == 4] = 43
    aseg[aseg == 5] = 44
    aseg[aseg == 7] = 46
    aseg[aseg == 8] = 47
    aseg[aseg == 10] = 49
    aseg[aseg == 11] = 50
    aseg[aseg == 12] = 51
    aseg[aseg == 13] = 52
    aseg[aseg == 17] = 53
    aseg[aseg == 18] = 54
    aseg[aseg == 26] = 58
    aseg[aseg == 28] = 60
    aseg[aseg == 31] = 63

    cortical_label_mask = (aseg >= 2000) & (aseg <= 2999)
    aseg[cortical_label_mask] = aseg[cortical_label_mask] - 1000

    labels_sag = np.array([0, 14, 15, 16, 24, 41, 43, 44, 46, 47, 49,
                           50, 51, 52, 53, 54, 58, 60, 63, 77, 251, 1001, 1002,
                           1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014,
                           1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025,
                           1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035])

    h, w, d = aseg.shape
    lut_aseg = np.zeros(max(labels_sag) + 1, dtype='int')
    for idx, value in enumerate(labels_sag):
        lut_aseg[value] = idx

    mapped_aseg_sag = lut_aseg.ravel()[aseg.ravel()]
    mapped_aseg_sag = mapped_aseg_sag.reshape((h, w, d))

    return mapped_aseg, mapped_aseg_sag


def map_wmparc2label(aseg):
    """
    将 wmparc 分割图的 FreeSurfer 标签映射到内部紧凑标签。

    和 map_aparc_aseg2label 类似，但针对白质分割（wmparc）做了调整：
    - 肼胝体 251~255 → 192（而非保留 251）
    - 包含白质高信号 77 的处理

    wmparc 是 FreeSurfer 中既包含皮层也包含白质的完整分割。
    """
    aseg = aseg.astype(np.int16)
    aseg_temp = aseg.copy()

    aseg[aseg == 80] = 77
    aseg[aseg == 85] = 0
    aseg[aseg == 62] = 41
    aseg[aseg == 30] = 2
    aseg[aseg == 72] = 24

    aseg[(aseg_temp >= 3000) & (aseg_temp < 3999)] = 2
    aseg[(aseg_temp >= 4000) & (aseg_temp < 4999)] = 41

    aseg[aseg_temp == 5001] = 2
    aseg[aseg_temp == 5002] = 41

    # 肼胝体合并为 192
    aseg[(aseg >= 251) & (aseg <= 255)] = 192

    aseg[aseg == 3] = 0
    aseg[aseg == 42] = 0

    if np.any(np.in1d([1000, 2000], aseg.ravel())):
        aseg = fill_unknown_labels_per_hemi(aseg, 1000, 2000)
        aseg = fill_unknown_labels_per_hemi(aseg, 2000, 3000)

    aseg = fill_WMhyper_per_hemi(aseg)

    cortical_label_mask = (aseg >= 2000) & (aseg <= 2999)
    aseg[cortical_label_mask] = aseg[cortical_label_mask] - 1000

    aseg[aseg_temp == 2014] = 2014
    aseg[aseg_temp == 2028] = 2028
    aseg[aseg_temp == 2012] = 2012
    aseg[aseg_temp == 2016] = 2016
    aseg[aseg_temp == 2002] = 2002
    aseg[aseg_temp == 2023] = 2023
    aseg[aseg_temp == 2017] = 2017
    aseg[aseg_temp == 2024] = 2024
    aseg[aseg_temp == 2010] = 2010
    aseg[aseg_temp == 2013] = 2013
    aseg[aseg_temp == 2025] = 2025
    aseg[aseg_temp == 2022] = 2022
    aseg[aseg_temp == 2021] = 2021
    aseg[aseg_temp == 2005] = 2005

    labels = np.array([0,  2,  4,  5,  7,  8,  10, 11, 12, 13, 14, 15,
                       16, 17, 18, 24, 26, 28, 31, 41, 43, 44, 46, 47,
                       49, 50, 51, 52, 53, 54, 58, 60, 63, 192,
                       1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012,
                       1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023,
                       1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035,
                       2002, 2005, 2010, 2012, 2013, 2014, 2016, 2017, 2021, 2022, 2023,
                       2024, 2025, 2028])

    h, w, d = aseg.shape
    lut_aseg = np.zeros(max(labels) + 1, dtype='int')
    for idx, value in enumerate(labels):
        lut_aseg[value] = idx

    mapped_aseg = lut_aseg.ravel()[aseg.ravel()]
    mapped_aseg = mapped_aseg.reshape((h, w, d))

    # 矢状面映射
    aseg[aseg == 2] = 41
    aseg[aseg == 3] = 42
    aseg[aseg == 4] = 43
    aseg[aseg == 5] = 44
    aseg[aseg == 7] = 46
    aseg[aseg == 8] = 47
    aseg[aseg == 10] = 49
    aseg[aseg == 11] = 50
    aseg[aseg == 12] = 51
    aseg[aseg == 13] = 52
    aseg[aseg == 17] = 53
    aseg[aseg == 18] = 54
    aseg[aseg == 26] = 58
    aseg[aseg == 28] = 60
    aseg[aseg == 31] = 63

    cortical_label_mask = (aseg >= 2000) & (aseg <= 2999)
    aseg[cortical_label_mask] = aseg[cortical_label_mask] - 1000

    labels_sag = np.array([0,  14, 15, 16, 24, 41, 43, 44, 46, 47, 49,
                           50, 51, 52, 53, 54, 58, 60, 63, 192,
                           1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013,
                           1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025,
                           1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035])

    h, w, d = aseg.shape
    lut_aseg = np.zeros(max(labels_sag) + 1, dtype='int')
    for idx, value in enumerate(labels_sag):
        lut_aseg[value] = idx

    mapped_aseg_sag = lut_aseg.ravel()[aseg.ravel()]
    mapped_aseg_sag = mapped_aseg_sag.reshape((h, w, d))

    return mapped_aseg, mapped_aseg_sag


def map_wmparc2gtseg(aseg):
    """
    将 wmparc 分割映射到最终 ground truth（不做紧凑索引化）。
    仅做标签合并和白质高信号去除，保留 FreeSurfer 标签值。
    """
    aseg = aseg.astype(np.int16)
    aseg_temp = aseg.copy()

    aseg[aseg == 80] = 77
    aseg[aseg == 85] = 0
    aseg[aseg == 62] = 41
    aseg[aseg == 30] = 2
    aseg[aseg == 72] = 24

    aseg[(aseg_temp >= 3000) & (aseg_temp < 3999)] = 2
    aseg[(aseg_temp >= 4000) & (aseg_temp < 4999)] = 41

    aseg[aseg_temp == 5001] = 2
    aseg[aseg_temp == 5002] = 41

    aseg[(aseg >= 251) & (aseg <= 255)] = 192

    aseg[aseg == 3] = 0
    aseg[aseg == 42] = 0

    if np.any(np.in1d([1000, 2000], aseg.ravel())):
        aseg = fill_unknown_labels_per_hemi(aseg, 1000, 2000)
        aseg = fill_unknown_labels_per_hemi(aseg, 2000, 3000)

    aseg = fill_WMhyper_per_hemi(aseg)

    return aseg


def sagittal_coronal_remap_lookup(x):
    """
    从左半球标签到对应右半球标签的查表函数。
    用于矢状/冠状网络中左右标签映射。
    """
    return {
        2: 41,
        3: 42,
        4: 43,
        5: 44,
        7: 46,
        8: 47,
        10: 49,
        11: 50,
        12: 51,
        13: 52,
        17: 53,
        18: 54,
        26: 58,
        28: 60,
        31: 63,
        }[x]


def map_prediction_sagittal2full(prediction_sag, num_classes=85):
    """
    将矢状网络的预测映射到完整的 FreeSurfer 标签空间。

    矢状网络使用的类别更少（54 vs 82），因为：
    - 左右半球在中矢状面上对称，许多左右标签被合并了
    - 但某些跨中线的结构（如脑干）仍保留左右区分

    这个函数通过一个重排序索引 (idx_list) 将矢状网络的
    54 类输出「展开」到 82 类的完整标签空间。
    """
    if num_classes == 96:
        idx_list = np.asarray([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 14, 15, 4, 16,
                               17, 18, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 20, 21, 22,
                               23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                               40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], dtype=np.int16)

    else:
        labels = np.array([0, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14,
                           15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44,
                           46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
                           192, 1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011,
                           1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022,
                           1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035,
                           2002, 2005, 2010, 2012, 2013, 2014, 2016, 2017, 2021, 2022, 2023,
                           2024, 2025, 2028])

        labels_full_to_sag = np.array([0, 41, 43, 44, 46, 47, 49, 50, 51, 52, 14, 15, 16, 53, 54, 24, 58, 60, 63, 41, 43, 44, 46, 47, 49, 50, 51, 52, 53,
                         54, 58, 60, 63, 192, 1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016,
                         1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035,
                         1002, 1005, 1010, 1012, 1013, 1014, 1016, 1017, 1021, 1022, 1023, 1024, 1025, 1028])

        labels_sag = np.array([0, 14, 15, 16, 24, 41, 43, 44, 46, 47, 49,
                               50, 51, 52, 53, 54, 58, 60, 63, 192, 1001, 1002,
                               1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014,
                               1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025,
                               1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035])

        # 对每个完整标签，找到其在矢状标签空间中的索引
        idx = []
        for l in labels_full_to_sag:
            idx.append(np.where(labels_sag==l)[0][0])

        idx_list = np.array(idx)

    # 重排通道维，使矢状面 logits 与轴状/冠状面的完整标签顺序对齐。
    # 这是三视角概率融合的关键步骤。
    prediction_full = prediction_sag[:, idx_list, :, :]
    return prediction_full


# =========================================================================
# 7. 连通域分析
# =========================================================================

def bbox_3d(img):
    """
    计算 3D 二值图像的包围盒（bounding box）。
    返回非零区域在 x, y, z 方向的最小/最大坐标。
    """
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax


def get_largest_cc(segmentation):
    """
    找到分割中最大的连通分量。
    常用于后处理：去除小的噪声连通块。

    :param segmentation: 二值分割 [H, W, D]
    :return: 只包含最大连通域的二值图
    """
    labels = label(segmentation, connectivity=3, background=0)

    bincount = np.bincount(labels.flat)
    background = np.argmax(bincount)
    bincount[background] = -1

    largest_cc = labels == np.argmax(bincount)

    return largest_cc


# =========================================================================
# 8. PyTorch Dataset 类
# =========================================================================

class OrigDataThickSlices(Dataset):
    """
    推理数据集（单模态版本）。

    对输入的 3D 体数据，按指定视角做轴置换，
    再构建厚切片，返回 2.5D 切片样本供模型推理。
    """
    def __init__(self, img_filename, orig, plane='Axial', slice_thickness=3, transforms=None):

        try:
            self.img_filename = img_filename
            self.plane = plane
            self.slice_thickness = slice_thickness

            # Transform Data as needed
            if plane == 'Sagittal':
                orig = transform_sagittal(orig)
                print('Loading Sagittal')
            elif plane == 'Axial':
                orig = transform_axial(orig)
                print('Loading Axial')
            else:
                print('Loading Coronal.')

            # Create Thick Slices
            orig_thick = get_thick_slices(orig, self.slice_thickness)

            # Make 4D: (D, H, W, C) — 切片维被提到第一个轴
            orig_thick = np.transpose(orig_thick, (2, 0, 1, 3))
            self.images = orig_thick

            self.count = self.images.shape[0]

            self.transforms = transforms

            print("Successfully loaded Image from {}".format(img_filename))

        except Exception as e:
            print("Loading failed. {}".format(e))

    def __getitem__(self, index):
        img = self.images[index]
        if self.transforms is not None:
            img = self.transforms(img)
        return {'image': img}

    def __len__(self):
        return self.count


class OrigDataThickSlices_Fused_Input(Dataset):
    """
    ★ 推理数据集（多模态融合版本）— DDSurfer_Pred.py 实际使用 ★

    输入：多个模态的 3D numpy 数组列表（如 FA/Trace/MinEig/MidEig）

    处理步骤：
    1. 每个模态按视角重排轴向
    2. 每个模态独立做厚切片（每个模态产生 7 通道的 2.5D 表示）
    3. 所有模态的厚切片在通道维拼接（4 模态 → 4×7=28 通道）

    输出：每次 __getitem__ 返回一个切片样本，供模型批量推理。

    为什么在通道维拼接而非 batch 维？
    - 这样每个 2D 卷积的输入是「多模态的上下文切片」
    - 网络可以在同一层同时看到 FA、Trace 等多种信息
    """
    def __init__(self, img_filename, orig, plane='Axial', slice_thickness=3, transforms=None):

        try:
            self.img_filename = img_filename
            self.plane = plane
            self.slice_thickness = slice_thickness

            # 每个模态先各自构建厚切片张量，再在通道维拼接。
            # 例如 4 模态时：每切片通道数 = 4 * 7 = 28。
            orig_thick_list = []
            for idx, orig_ in enumerate(orig):
                # Transform Data as needed
                if plane == 'Sagittal':
                    orig_ = transform_sagittal(orig_)
                    print('Loading Sagittal %d' % idx)
                elif plane == 'Axial':
                    orig_ = transform_axial(orig_)
                    print('Loading Axial %d' % idx)
                else:
                    print('Loading Coronal %d' % idx)

                # Create Thick Slices
                orig_thick = get_thick_slices(orig_, self.slice_thickness)

                # Make 4D: (D, H, W, C)
                orig_thick = np.transpose(orig_thick, (2, 0, 1, 3))
                orig_thick_list.append(orig_thick)

            # 送入 ToTensorTest 前，每个切片形状为：H × W × (模态数 × 7)。
            # ToTensorTest 会把它转成 C × H × W（PyTorch 格式）。
            self.images = np.concatenate(orig_thick_list, axis=3)

            self.count = self.images.shape[0]

            self.transforms = transforms

            print("Successfully loaded Image from {}".format(img_filename))

        except Exception as e:
            print("Loading failed. {}".format(e))

    def __getitem__(self, index):
        img = self.images[index]
        if self.transforms is not None:
            img = self.transforms(img)
        return {'image': img}

    def __len__(self):
        return self.count


# =========================================================================
# 训练数据集类
# =========================================================================

class AsegDatasetWithAugmentation(Dataset):
    """
    训练数据集（单模态）：从 hdf5 文件加载厚切片样本。

    hdf5 文件中包含：
    - orig_dataset:  厚切片图像数据
    - aseg_dataset:  分割标签
    - weight_dataset: 权重图（中位数频率平衡 + 边缘加权）
    - subject:       被试 ID
    """
    def __init__(self, params, transforms=None):
        try:
            self.params = params

            with h5py.File(self.params['dataset_name'], "r") as hf:
                self.images = np.array(hf.get('orig_dataset'))
                self.labels = np.array(hf.get('aseg_dataset'))
                self.weights = np.array(hf.get('weight_dataset'))
                self.subjects = np.array(hf.get("subject"))

            self.count = self.images.shape[0]
            self.transforms = transforms

            print("Successfully loaded {} with plane: {}".format(params["dataset_name"], params["plane"]))

        except Exception as e:
            print("Loading failed: {}".format(e))

    def get_subject_names(self):
        return self.subjects

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        weight = self.weights[index]

        if self.transforms is not None:
            tx_sample = self.transforms({'img': img, 'label': label, 'weight': weight})
            img = tx_sample['img']
            label = tx_sample['label']
            weight = tx_sample['weight']

        return {'image': img, 'label': label, 'weight': weight}

    def __len__(self):
        return self.count


class AsegDatasetWithAugmentation_Fused_Input(Dataset):
    """
    训练数据集（多模态融合版本）：从多个 hdf5 文件加载，
    在通道维拼接各模态的厚切片。
    """
    def __init__(self, params, transforms=None):
        try:
            self.params = params

            self.images = []
            for idx in range(len(self.params['dataset_name'])):
                with h5py.File(self.params['dataset_name'][idx], "r") as hf:
                    self.images.append(np.array(hf.get('orig_dataset')))
                    if idx == 0:
                        self.labels = np.array(hf.get('aseg_dataset'))
                        self.weights = np.array(hf.get('weight_dataset'))
                        self.subjects = np.array(hf.get("subject"))

            self.images = np.concatenate(self.images, axis=3)
            self.count = self.images.shape[0]
            self.transforms = transforms

            print("Successfully loaded {} with plane: {}".format(params["dataset_name"], params["plane"]))

        except Exception as e:
            print("Loading failed: {}".format(e))

    def get_subject_names(self):
        return self.subjects

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        weight = self.weights[index]

        if self.transforms is not None:
            tx_sample = self.transforms({'img': img, 'label': label, 'weight': weight})
            img = tx_sample['img']
            label = tx_sample['label']
            weight = tx_sample['weight']

        return {'image': img, 'label': label, 'weight': weight}

    def __len__(self):
        return self.count


class AsegDatasetWithAugmentation_Slice(Dataset):
    """
    训练数据集（单模态，每个切片单独存为一个 hdf5 文件）。
    适用于每个切片单独预处理后存储的情况。
    """
    def __init__(self, params, transforms=None):
        try:
            self.params = params
            self.hdf5_files = sorted(glob.glob(os.path.join(params['dataset_name'], '*hdf5')))
            self.count = len(self.hdf5_files)
            self.transforms = transforms
            print("Successfully loaded {} with plane: {}".format(params["dataset_name"], params["plane"]))
        except Exception as e:
            print("Loading failed: {}".format(e))

    def get_subject_names(self):
        return self.subjects

    def __getitem__(self, index):
        hdf5 = self.hdf5_files[index]
        with h5py.File(hdf5, "r") as hf:
            img = np.array(hf.get('orig_dataset')).squeeze()
            label = np.array(hf.get('aseg_dataset')).squeeze()
            weight = np.array(hf.get('weight_dataset')).squeeze()

        if self.transforms is not None:
            tx_sample = self.transforms({'img': img, 'label': label, 'weight': weight})
            img = tx_sample['img']
            label = tx_sample['label']
            weight = tx_sample['weight']

        return {'image': img, 'label': label, 'weight': weight}

    def __len__(self):
        return self.count


class AsegDatasetWithAugmentation_Slice_Fused_Input(Dataset):
    """
    训练数据集（多模态融合，每个切片单独存为 hdf5 文件）。
    从多个模态目录中各取对应切片文件，拼接后返回。
    """
    def __init__(self, params, transforms=None):
        try:
            self.params = params
            self.params['dataset_name'] = sorted(self.params['dataset_name'])
            self.hdf5_files = []
            tmp_len = []
            for dataset in self.params['dataset_name']:
                filelist = sorted(glob.glob(os.path.join(dataset, '*hdf5')))
                self.hdf5_files.append(filelist)
                tmp_len.append(len(filelist))

            if np.unique(tmp_len).shape[0] != 1:
                print("Error: Input data should have the same number of hdf5 files!")
                exit()

            self.hdf5_files = np.array(self.hdf5_files)
            self.count = np.unique(tmp_len)[0]
            self.transforms = transforms

            print("Successfully loaded {} with plane: {}".format(params["dataset_name"], params["plane"]))

        except Exception as e:
            print("Loading failed: {}".format(e))

    def get_subject_names(self):
        return self.subjects

    def __getitem__(self, index):
        hdf5s = self.hdf5_files[:, index]
        if np.unique([os.path.split(n)[1] for n in hdf5s]).shape[0] != 1:
            print("Error: Must have the same file name!")
            print(hdf5s)
            exit()

        img = []
        for idx, hdf5 in enumerate(hdf5s):
            with h5py.File(hdf5, "r") as hf:
                img.append(np.array(hf.get('orig_dataset')).squeeze())
                if idx == 0:
                    label = np.array(hf.get('aseg_dataset')).squeeze()
                    weight = np.array(hf.get('weight_dataset')).squeeze()
        img = np.concatenate(img, axis=2)

        if self.transforms is not None:
            tx_sample = self.transforms({'img': img, 'label': label, 'weight': weight})
            img = tx_sample['img']
            label = tx_sample['label']
            weight = tx_sample['weight']

        return {'image': img, 'label': label, 'weight': weight}

    def __len__(self):
        return self.count

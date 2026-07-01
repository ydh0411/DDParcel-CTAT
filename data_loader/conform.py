
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
# conform.py — MRI 影像空间一致性化
# =========================================================================
# 这个文件来自 FastSurfer 项目，用于将任意来源的 MRI 脑图像
# 统一到标准格式，使得深度学习网络可以处理。
#
# === 什么是 conform？===
# 来自不同扫描仪/医院/研究的 MRI 可能有不同的：
# - 体素大小（0.8mm vs 1.2mm vs 非各向同性）
# - 图像尺寸（256×256×180 vs 320×320×256）
# - 空间方向（LAS vs RAS vs LPS 等）
#
# conform 就是把它们全部统一到：
# - 尺寸：256 × 256 × 256
# - 体素：1mm × 1mm × 1mm（各向同性）
# - 方向：LIA（FreeSurfer 标准方向，本质上是 RAS 的一种排列）
# - 类型：float32
#
# === 类比 ===
# 就像把所有照片统一裁剪为 256×256 像素、RGB 格式。
# 这样网络就不需要学习处理不同大小/分辨率的输入了。
# =========================================================================
import optparse
import sys
import numpy as np
import nibabel as nib

HELPTEXT = """
Script to conform an MRI brain image to UCHAR, RAS orientation, and 1mm isotropic voxels


USAGE:
conform.py  -i <input> -o <output>


Dependencies:
    Python 3.5

    Numpy
    http://www.numpy.org

    Nibabel to read and write FreeSurfer data
    http://nipy.org/nibabel/


Original Author: Martin Reuter
Date: Jul-09-2019

"""

h_input = 'path to input image'
h_output = 'path to ouput image'
h_order = 'order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)'


def options_parse():
    """
    Command line option parser
    """
    parser = optparse.OptionParser(version='$Id: conform.py,v 1.0 2019/07/19 10:52:08 mreuter Exp $',
                                   usage=HELPTEXT)
    parser.add_option('--input', '-i', dest='input', help=h_input)
    parser.add_option('--output', '-o', dest='output', help=h_output)
    parser.add_option('--order', dest='order', help=h_order, type="int", default=1)
    (fin_options, args) = parser.parse_args()
    if fin_options.input is None or fin_options.output is None:
        sys.exit('ERROR: Please specify input and output images')
    return fin_options


# =========================================================================
# 核心函数：map_image — 重采样到目标空间
# =========================================================================
def map_image(img, out_affine, out_shape, ras2ras=np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
              order=1):
    """
    将图像从原始空间映射到目标空间。

    技术上通过 scipy.ndimage.affine_transform 实现：
    1. 计算原始体素空间 → 目标体素空间的变换矩阵（vox2vox）
    2. 在目标网格上逐点采样原始图像的值
    3. 插值方法由 order 指定（0=最近邻，1=线性，...）

    :param nibabel.MGHImage img: 原始 3D 图像
    :param np.ndarray out_affine: 目标图像的仿射矩阵
    :param np.ndarray out_shape: 目标图像的形状 [256, 256, 256]
    :param np.ndarray ras2ras: 额外的 RAS→RAS 变换（默认恒等）
    :param int order: 插值阶数
    :return: 重采样后的图像数据数组
    """
    from scipy.ndimage import affine_transform
    from numpy.linalg import inv

    # 计算体素到体素的变换
    # out_affine^{-1} @ ras2ras @ img.affine
    # 这个变换告诉我们在原始体素索引(x,y,z)和目标体素索引(x',y',z')之间的关系
    vox2vox = inv(out_affine) @ ras2ras @ img.affine

    # 应用逆变换（从目标拉回原始空间采样）
    # 这就是重采样的本质：在目标网格的每个位置，
    # 用变换找到它在原图中的对应位置，然后插值
    new_data = affine_transform(img.get_fdata(), inv(vox2vox), output_shape=out_shape, order=order)
    return new_data


# =========================================================================
# getscale — 计算强度缩放参数
# =========================================================================
def getscale(data, dst_min, dst_max, f_low=0.0, f_high=0.999):
    """
    计算图像强度缩放参数（类似 FreeSurfer 的 mri_convert）。

    getscale 找到需要裁剪的低端百分位和高端的 0.1% 分位，
    然后用线性缩放将数据映射到 [dst_min, dst_max]。

    为什么要做这个缩放？
    - 不同扫描的影像强度范围差异很大
    - 缩放到统一范围 [0, 255] 有助于网络训练
    - 使用 [0.0, 0.999] 的百分位裁剪可以去除极端异常值

    :param np.ndarray data: 图像强度数据
    :param float dst_min: 目标最小值（通常 0）
    :param float dst_max: 目标最大值（通常 255）
    :param f_low: 低端裁剪百分位（0.0=不做裁剪）
    :param f_high: 高端裁剪百分位（0.999=裁剪千分之一的高值）
    :return: (src_min, scale) 偏移量和缩放因子
    """
    data = np.clip(data, a_min=0, a_max=np.max(data))

    src_min = np.min(data)
    src_max = np.max(data)

    if src_min < 0.0:
        sys.exit('ERROR: Min value in input is below 0.0!')

    print("Input:    min: " + format(src_min) + "  max: " + format(src_max))

    if f_low == 0.0 and f_high == 1.0:
        return src_min, 1.0

    nz = (np.abs(data) >= 1e-15).sum()
    voxnum = data.shape[0] * data.shape[1] * data.shape[2]

    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, bin_edges = np.histogram(data, histosize)

    cs = np.concatenate(([0], np.cumsum(hist)))

    # 找到低端第 f_low 百分位
    nth = int(f_low * voxnum)
    idx = np.where(cs < nth)
    if len(idx[0]) > 0:
        idx = idx[0][-1] + 1
    else:
        idx = 0
    src_min = idx * bin_size + src_min

    # 找到高端第 f_high 百分位
    nth = voxnum - int((1.0 - f_high) * nz)
    idx = np.where(cs >= nth)
    if len(idx[0]) > 0:
        idx = idx[0][-2]
    else:
        print('ERROR: rescale upper bound not found')
    src_max = idx * bin_size + src_min

    if src_min == src_max:
        scale = 1.0
    else:
        scale = (dst_max - dst_min) / (src_max - src_min)

    print("rescale:  min: " + format(src_min) + "  max: " + format(src_max) + "  scale: " + format(scale))

    return src_min, scale


# =========================================================================
# scalecrop — 应用缩放和裁剪
# =========================================================================
def scalecrop(data, dst_min, dst_max, src_min, scale):
    """
    按 getscale 计算的参数对数据做缩放。

    new_data = dst_min + scale * (data - src_min)
    然后裁剪到 [dst_min, dst_max]
    """
    data_new = dst_min + scale * (data - src_min)
    data_new = np.clip(data_new, dst_min, dst_max)
    print("Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max()))
    return data_new


# =========================================================================
# rescale — 完整缩放到目标范围（一次性调用 getscale + scalecrop）
# =========================================================================
def rescale(data, dst_min, dst_max, f_low=0.0, f_high=0.999):
    """
    鲁棒性地将图像强度缩放到 [dst_min, dst_max]。
    内部调用 getscale 计算参数，再调用 scalecrop 应用。
    """
    src_min, scale = getscale(data, dst_min, dst_max, f_low, f_high)
    data_new = scalecrop(data, dst_min, dst_max, src_min, scale)
    return data_new


# =========================================================================
# ★★★ conform — 最核心函数 ★★★
# =========================================================================
def conform(img, order=1, rescale=False, imagetype='image'):
    """
    Python 实现的 mri_convert -c。

    主要步骤：
    1. 创建标准的 256×256×256 网格，1mm 各向同性
    2. 用 map_image 将原始图像重采样到这个网格
    3. 将背景区域设为 -4.0（和 FreeSurfer 的约定一致）
    4. 可选：将强度缩放到 [0, 255]

    背景设为 -4 的原因：
    - FastSurfer 在训练时对脑掩膜外的体素统一设为 -4
    - 这样网络可以明确区分「脑组织」和「非脑组织」
    - normalize.py 中也对掩膜外的体素设为 -4，保持一致

    :param nibabel.MGHImage img: 加载的源图像
    :param int order: 插值阶数（0=最近邻，1=线性，...）
    :param bool rescale: 是否缩放到 [0, 255]（默认 False）
    :param imagetype: 'image' 或 'label'
    :return: nibabel.MGHImage：统一后的图像
    """
    from nibabel.freesurfer.mghformat import MGHHeader

    cwidth = 256
    csize = 1
    # 从原图 header 创建新 header
    h1 = MGHHeader.from_header(img.header)

    # 设置标准参数
    h1.set_data_shape([cwidth, cwidth, cwidth, 1])     # 256³
    h1.set_zooms([csize, csize, csize])                   # 1mm 各向同性
    h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]     # LIA 方向矩阵
    h1['fov'] = cwidth
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]  # 保持物理空间中心位置

    # 重采样图像到标准网格
    mapped_data = map_image(img, h1.get_affine(), h1.get_data_shape(), order=order)

    # 将背景设为 -4
    if imagetype == 'image':
        # 找到最大的空腔区域（通常是图像外的背景）
        background = get_largest_cc(mapped_data == 0)
        if background[0,0,0]:
            mapped_data[background == 1] = -4.0
        else:
            mapped_data[background == 0] = -4.0

    # 可选强度缩放
    if rescale:
        src_min, scale = getscale(img.get_data(), 0, 255)
        mapped_data = scalecrop(mapped_data, 0, 255, src_min, scale)

    new_data = np.float32(mapped_data)
    new_data[np.isnan(new_data)] = 0.0
    new_img = nib.MGHImage(new_data, h1.get_affine(), h1)

    new_img.set_data_dtype(np.float32)

    return new_img


# =========================================================================
# 辅助函数
# =========================================================================

def get_largest_cc(segmentation):
    """
    找到二值分割中最大的连通分量。
    在 conform 中用于找到「背景」区域。
    """
    from skimage.measure import label
    labels = label(segmentation, connectivity=3, background=0)

    bincount = np.bincount(labels.flat)
    background = np.argmax(bincount)
    bincount[background] = -1

    largest_cc = labels == np.argmax(bincount)

    return largest_cc


# =========================================================================
# is_conform — 判断图像是否已经标准化
# =========================================================================
def is_conform(img, eps=1e-06):
    """
    检查图像是否已经符合标准（已经 conform 过）。

    检查三个条件：
    1. 尺寸是否为 256×256×256
    2. 体素是否 1mm 各向同性
    3. 方向是否为 LIA

    :param nibabel.MGHImage img: 加载的图像
    :param float eps: LIA 方向检查的容差
    :return: True=已经 conform，False=需要 conform
    """
    ishape = img.shape

    if len(ishape) > 3 and ishape[3] != 1:
        sys.exit('ERROR: Multiple input frames (' + format(img.shape[3]) + ') not supported!')

    # 检查尺寸
    if ishape[0] != 256 or ishape[1] != 256 or ishape[2] != 256:
        return False

    # 检查体素大小
    izoom = img.header.get_zooms()
    if izoom[0] != 1.0 or izoom[1] != 1.0 or izoom[2] != 1.0:
        return False

    # 检查 LIA 方向
    iaffine = img.affine[0:3, 0:3] + [[1, 0, 0], [0, 0, -1], [0, 1, 0]]

    if np.max(np.abs(iaffine)) > 0.0 + eps:
        return False

    return True


# =========================================================================
# check_affine_in_nifti — 检查 NIfTI 仿射矩阵一致性
# =========================================================================
def check_affine_in_nifti(img, logger=None):
    """
    检查 NIfTI 图像中仿射矩阵的一致性。

    NIfTI 格式有两个仿射矩阵：qform 和 sform。
    理想情况下它们应该一致。如果不一致，优先使用 qform。
    如果体素尺寸在 header 和 affine 中不一致，则返回错误。

    :param nibabel.NiftiImage img: 加载的 NIfTI 图像
    :return: True=一致性检查通过，False=存在不一致
    """
    check = True
    message = ""

    if img.header['qform_code'] != 0 and np.max(np.abs(img.get_sform() - img.get_qform())) > 0.001:
        # qform 和 sform 不一致，使用 qform 替代
        message = ' # qform and sform transform are not identical!'
        img.set_sform(img.get_qform())
        img.update_header()

    else:
        # 检查仿射矩阵中的体素尺寸和 header 是否一致
        vox_size_head = img.header.get_zooms()
        aff = img.affine
        xsize = np.sqrt(aff[0][0] * aff[0][0] + aff[1][0] * aff[1][0] + aff[2][0] * aff[2][0])
        ysize = np.sqrt(aff[0][1] * aff[0][1] + aff[1][1] * aff[1][1] + aff[2][1] * aff[2][1])
        zsize = np.sqrt(aff[0][2] * aff[0][2] + aff[1][2] * aff[1][2] + aff[2][2] * aff[2][2])

        if (abs(xsize - vox_size_head[0]) > .001) or (abs(ysize - vox_size_head[1]) > .001) or (abs(zsize - vox_size_head[2]) > 0.001):
            message = "#############################################################\n" \
                      "ERROR: Invalid Nifti-header! Affine matrix is inconsistent with Voxel sizes. " \
                      "\nVoxel size (from header) vs. Voxel size in affine: " \
                      "({}, {}, {}), ({}, {}, {})\nInput Affine----------------\n{}\n" \
                      "#############################################################".format(vox_size_head[0],
                                                                                             vox_size_head[1],
                                                                                             vox_size_head[2],
                                                                                             xsize, ysize, zsize,
                                                                                             aff)
            check = False

    if logger is not None:
        logger.info(message)
    else:
        print(message)

    return check


if __name__ == "__main__":
    # Command Line options are error checking done here
    options = options_parse()

    print("Reading input: {} ...".format(options.input))
    image = nib.load(options.input)

    if len(image.shape) > 3 and image.shape[3] != 1:
        sys.exit('ERROR: Multiple input frames (' + format(image.shape[3]) + ') not supported!')

    if is_conform(image):
        sys.exit("Input " + format(options.input) + " is already conform! No output created.\n")

    # If image is nifti image
    if options.input[-7:] == ".nii.gz" or options.input[-4:] == ".nii":
        if not check_affine_in_nifti(image):
            sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")

    new_image = conform(image, options.order)
    print ("Writing conformed image: {}".format(options.output))

    nib.save(new_image, options.output)

    sys.exit(0)

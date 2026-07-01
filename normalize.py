# =========================================================================
# normalize.py — DTI 标量图预处理
# =========================================================================
# 这个工具是 DDParcel 预处理流水线中的第 7 步（参见 process.sh）。
# 它在脑掩膜内对每个 DTI 标量图做 z-score 标准化，并将背景设为固定值 -4。
#
# === 为什么要做标准化？===
# 不同被试的 FA/Trace/MinEig/MidEig 值范围不同。
# 如果不标准化，网络可能会把「值大」和「结构重要」混淆。
# z-score 让不同被试的同一模态在相似的数值范围内，减少被试间差异。
#
# === 为什么背景设为 -4？===
# 网络在训练时看到背景区域的像素值恒定是 -4，
# 这给网络提供了明确的「这里是背景，请忽略」的信号。
# 同时 -4 是一个远离正常脑组织 z-score 范围（通常在 -3~5 之间）的值，
# 不容易和真实组织混淆。
# =========================================================================
import argparse
import nibabel as nib
import numpy as np
import scipy.stats as stats
import os


def main():
    # =========================================================================
    # 命令行参数解析
    # =========================================================================
    parser = argparse.ArgumentParser(
        description="",
        epilog="")
    parser.add_argument("-v", "--version",
                        action="version", default=argparse.SUPPRESS,
                        version='1.0',
                        help="Show program's version number and exit")
    parser.add_argument('--input', default="", help='input DTI scalar volume (FA/Trace/MinEig/MidEig)')
    parser.add_argument('--mask', default="", help='brain mask file')
    parser.add_argument('--output', default="", help='output normalized file')
    parser.add_argument('--flip', type=int, default=0, help='flip mask along up-down (1) or left-right (2)')

    args = parser.parse_args()

    # ──────────── 加载数据 ────────────
    img = nib.load(args.input)
    img_data = img.get_fdata()
    img_affine = img.affine
    img_header = img.header

    # 如果未提供 mask，则使用输入自身作为 mask（自掩膜模式）
    if not os.path.exists(args.mask):
        args.mask = args.input

    mask = nib.load(args.mask)
    mask_data = mask.get_fdata()

    # ──────────── 翻转掩膜（可选） ────────────
    # HCP 数据常常需要上下翻转（flip=1），
    # 因为 HCP 数据的存储方向（neurological convention）
    # 和 nibabel 默认的读取方式可能有差异，
    # 导致 mask 和 DTI 数据的上下方向相反。
    if args.flip == 1:
        mask_data = np.flipud(mask_data)
    if args.flip == 2:
        mask_data = np.fliplr(mask_data)

    # ──────────── 掩膜二值化 ────────────
    # 将掩膜转成严格的 0/1 二值：
    # mask > 0 → 1（脑内）
    # mask ≤ 0 → 0（脑外/背景）
    mask_data[mask_data > 0] = 1
    mask_data[mask_data <= 0] = 0

    # 脑外的体素设为 NaN，这样它们在统计计算中会被自动忽略
    img_data[mask_data == 0] = np.nan

    # ═══════════════════════════════════════════════════════════════
    # 模态特定的范围裁剪（经验阈值）
    # ═══════════════════════════════════════════════════════════════
    # 为什么要裁剪范围？
    # 尽管 z-score 理论上对异常值鲁棒，但如果极少量体素的 DTI 值
    # 严重偏离（如 100 倍于正常范围），它们的 z-score 会非常大，
    # 导致标准化后的整体分布偏移。
    #
    # 这些阈值来自训练数据的经验观察：
    # - 特征值（Eigenvalue）正常情况下不会超过 0.004
    # - Trace（总扩散率）正常情况下不超过 0.012
    # - FA（各向异性分数）理论上在 [0, 1] 之间
    #
    # 注意：这里只做裁剪（clip），不做缩放。
    # 裁剪只限制极值，保留相对关系，z-score 在后面做。
    if args.input.find("dti-MaxEigenvalue") > 0 or args.input.find("dti-MidEigenvalue") > 0 or args.input.find("dti-MinEigenvalue") > 0:
        print("%d voxels are outside the expected range." % (np.sum(img_data[mask_data == 1] > 0.004) + np.sum(img_data[mask_data == 1] < 0.0)))
        img_data[mask_data == 1] = np.clip(img_data[mask_data == 1], a_min=0, a_max=0.004)

    elif args.input.find("dti-Trace") > 0:
        print("%d voxels are outside the expected range." % (np.sum(img_data[mask_data == 1] > 0.012) + np.sum(img_data[mask_data == 1] < 0.0)))
        img_data[mask_data == 1] = np.clip(img_data[mask_data == 1], a_min=0, a_max=0.012)

    elif args.input.find("dti-FractionalAnisotropy") > 0:
        print("%d voxels are outside the expected range." % (np.sum(img_data[mask_data == 1] > 1.0) + np.sum(img_data[mask_data == 1] < 0.0)))
        img_data[mask_data == 1] = np.clip(img_data[mask_data == 1], a_min=0, a_max=1)

    # ═══════════════════════════════════════════════════════════════
    # z-score 标准化（仅脑内体素）
    # ═══════════════════════════════════════════════════════════════
    # z = (x - μ) / σ
    # 其中 μ 和 σ 只基于脑内体素计算（背景的 NaN 被自动忽略）
    # 标准化后脑内体素分布为 μ=0, σ=1
    img_data[mask_data == 1] = stats.zscore(img_data[mask_data == 1])

    # 背景设为 -4（和 FastSurfer conform 中的约定保持一致）
    img_data[mask_data == 0] = -4.0

    print("z score: %f - %f " % (np.nanmin(img_data), np.nanmax(img_data)))

    # ──────────── 保存输出 ────────────
    NormMasked = nib.Nifti1Image(img_data, affine=img_affine, header=img_header)
    nib.save(NormMasked, args.output)


if __name__ == '__main__':
    main()

#!/usr/bin/env bash
# DDParcel 端到端流程：
# 原始 DWI/bval/bvec/mask -> DTI 标量图 -> 配准到 atlas -> 归一化 -> DDSurfer 分割。
#
# 注意：
# - 该脚本支持断点续跑：每个阶段都会先检查输出文件是否已存在。
# - "$1" 是可选命令前缀（例如 "python"、"singularity exec ..."，也可为空）。
#   保留该设计是为了兼容原有使用方式。
#
# 初学者阅读提示（强烈建议）：
# - 这不是“训练脚本”，而是“推理流水线脚本”。
# - 你可以把它理解成 9 个连续步骤，每步都把输入文件加工成新的输出文件。
# - 如果中途失败，修复后直接重跑 `bash process.sh`，它会从缺失步骤继续。

# 1) 定位所需 Slicer CLI 可执行文件（macOS 直接调用；Linux 通过 --launch 包装）。
if [ -x /Applications/Slicer.app/Contents/MacOS/Slicer ]; then
	SLICER_DMRI_DIR=$(dirname "$(find /Applications/Slicer.app/Contents -path "*/SlicerDMRI/lib/Slicer-*/cli-modules/DWIToDTIEstimation" -type f 2>/dev/null | head -n 1)")
	SLICER_CORE_DIR=$(dirname "$(find /Applications/Slicer.app/Contents -path "*/lib/Slicer-*/cli-modules/BRAINSFit" -type f 2>/dev/null | head -n 1)")
	DWIToDTIEstimation="$SLICER_DMRI_DIR/DWIToDTIEstimation"
	DiffusionTensorScalarMeasurements="$SLICER_DMRI_DIR/DiffusionTensorScalarMeasurements"
	BRAINSFit="$SLICER_CORE_DIR/BRAINSFit"
	ResampleScalarVectorDWIVolume="$SLICER_CORE_DIR/ResampleScalarVectorDWIVolume"
else
	SLICER_BIN=/data01/software/slicer/Slicer-5.2.2-linux-amd64/Slicer
	SLICER_DMRI_DIR=/data01/software/slicer/Slicer-5.2.2-linux-amd64/NA-MIC/Extensions-31382/SlicerDMRI/lib/Slicer-5.2/cli-modules
	SLICER_CORE_DIR=/data01/software/slicer/Slicer-5.2.2-linux-amd64/lib/Slicer-5.2/cli-modules
	DWIToDTIEstimation="$SLICER_BIN --launch $SLICER_DMRI_DIR/DWIToDTIEstimation"
	DiffusionTensorScalarMeasurements="$SLICER_BIN --launch $SLICER_DMRI_DIR/DiffusionTensorScalarMeasurements"
	BRAINSFit="$SLICER_BIN --launch $SLICER_CORE_DIR/BRAINSFit"
	ResampleScalarVectorDWIVolume="$SLICER_BIN --launch $SLICER_CORE_DIR/ResampleScalarVectorDWIVolume"
fi

atlas_T2=./100HCP-population-mean-T2-1mm.nii.gz

subID=HCP-100337-b1000

inputdir=testdata
outputdir=$inputdir/$subID/
mkdir -p $outputdir

# 2) 声明原始输入文件。
#    这里假设你的 testdata 目录里有：
#    - HCP-100337-b1000.nii.gz      (DWI 4D 影像)
#    - HCP-100337-b1000.bval/.bvec  (扩散梯度信息)
#    - HCP-100337-b1000-mask.nii.gz (脑掩膜)
dwi=$inputdir/$subID.nii.gz
bval=$inputdir/$subID.bval
bvec=$inputdir/$subID.bvec
mask=$inputdir/$subID-mask.nii.gz

nrrd_dwi=$inputdir/$subID.nhdr
nrrd_mask=$inputdir/$subID-mask.nhdr
# 3) 将 NIfTI 输入转换为 NHDR/NRRD，供 Slicer DMRI CLI 使用。
if [ ! -f  $nrrd_mask ]; then
	$1 nhdr_write.py --nifti $dwi --bval $bval --bvec $bvec --nhdr $nrrd_dwi
	$1 nhdr_write.py --nifti $mask --nhdr $nrrd_mask
fi

# 4) 从原始 DWI 估计扩散张量和 b0 图像。
#    输出：
#    - *-dti.nhdr  : 扩散张量场
#    - *-b0.nhdr   : b0 参考图（后续做配准）
nrrd_dti=$outputdir/$subID-dti.nhdr
nrrd_b0=$outputdir/$subID-b0.nhdr
if [ ! -f $nrrd_b0 ]; then
    $1 $DWIToDTIEstimation --enumeration LS $nrrd_dwi $nrrd_dti $nrrd_b0 #-m $nrrd_mask
fi

nrrd_fa=$outputdir/$subID-dti-FractionalAnisotropy.nhdr
nrrd_trace=$outputdir/$subID-dti-Trace.nhdr
nrrd_minEig=$outputdir/$subID-dti-MinEigenvalue.nhdr
nrrd_midEig=$outputdir/$subID-dti-MidEigenvalue.nhdr

nii_fa=$outputdir/$subID-dti-FractionalAnisotropy.nii.gz
nii_trace=$outputdir/$subID-dti-Trace.nii.gz
nii_minEig=$outputdir/$subID-dti-MinEigenvalue.nii.gz
nii_midEig=$outputdir/$subID-dti-MidEigenvalue.nii.gz
 
if [ ! -f $nii_midEig ]; then
    # 5) 从 DTI 张量提取标量图（FA/Trace/MinEig/MidEig），
    # 再转回 NIfTI 供后续配准和深度学习使用。
    #
    # 四个模态的作用（简化理解）：
    # - FA: 各向异性强度
    # - Trace: 总扩散量
    # - MinEig/MidEig: 张量特征值（表征扩散方向特性）
    $1 $DiffusionTensorScalarMeasurements --enumeration FractionalAnisotropy $nrrd_dti $nrrd_fa
    $1 $DiffusionTensorScalarMeasurements --enumeration Trace $nrrd_dti $nrrd_trace
    $1 $DiffusionTensorScalarMeasurements --enumeration MinEigenvalue $nrrd_dti $nrrd_minEig
    $1 $DiffusionTensorScalarMeasurements --enumeration MidEigenvalue $nrrd_dti $nrrd_midEig

    $1 nifti_write.py -i $nrrd_fa     -p ${nrrd_fa//.nhdr/}
    $1 nifti_write.py -i $nrrd_trace  -p ${nrrd_trace//.nhdr/}
    $1 nifti_write.py -i $nrrd_minEig -p ${nrrd_minEig//.nhdr/}
    $1 nifti_write.py -i $nrrd_midEig -p ${nrrd_midEig//.nhdr/}
fi

# 6) 通过 b0 -> atlas 变换，将被试扩散数据配准到 atlas 空间。
#    为什么先配准再喂网络？
#    - 训练时网络看到的是统一空间（256^3、方向一致）的数据；
#    - 推理时也要对齐到同一参考空间，标签才有可比性。
tfm=$outputdir/$subID-b0ToAtlasT2.tfm 
tfminv=$outputdir/$subID-b0ToAtlasT2_Inverse.h5
if [ ! -f $tfm ]; then
	$1 $BRAINSFit --fixedVolume $atlas_T2 --movingVolume $nrrd_b0 --linearTransform $outputdir/$subID-b0ToAtlasT2.tfm --useRigid --useAffine
fi

nii_fa_reg=$outputdir/$subID-dti-FractionalAnisotropy-Reg.nii.gz
nii_trace_reg=$outputdir/$subID-dti-Trace-Reg.nii.gz
nii_minEig_reg=$outputdir/$subID-dti-MinEigenvalue-Reg.nii.gz
nii_midEig_reg=$outputdir/$subID-dti-MidEigenvalue-Reg.nii.gz
nii_mask_reg=$outputdir/$subID-mask-Reg.nii.gz
if [ ! -f $nii_mask_reg ]; then
	# 对所有标量图和 mask 应用同一变换。
	$1 $ResampleScalarVectorDWIVolume -i linear ${nii_fa}     --Reference ${atlas_T2}     --transformationFile $tfm $nii_fa_reg
	$1 $ResampleScalarVectorDWIVolume -i linear ${nii_trace}  --Reference ${atlas_T2}  --transformationFile $tfm $nii_trace_reg
	$1 $ResampleScalarVectorDWIVolume -i linear ${nii_minEig} --Reference ${atlas_T2} --transformationFile $tfm $nii_minEig_reg
	$1 $ResampleScalarVectorDWIVolume -i linear ${nii_midEig} --Reference ${atlas_T2} --transformationFile $tfm $nii_midEig_reg
	$1 $ResampleScalarVectorDWIVolume -i nn     ${mask}       --Reference ${atlas_T2}       --transformationFile $tfm $nii_mask_reg
fi


# 7) 在脑掩膜内对各标量图做 z-score 归一化，并将背景设为 -4。
#    输出文件名后缀 `-Reg-NormMasked.nii.gz` 非常关键，
#    DDSurfer_Pred.py 就是按这个后缀去自动收集输入。
nii_fa_reg_norm=$outputdir/$subID-dti-FractionalAnisotropy-Reg-NormMasked.nii.gz
nii_trace_reg_norm=$outputdir/$subID-dti-Trace-Reg-NormMasked.nii.gz
nii_minEig_reg_norm=$outputdir/$subID-dti-MinEigenvalue-Reg-NormMasked.nii.gz
nii_midEig_reg_norm=$outputdir/$subID-dti-MidEigenvalue-Reg-NormMasked.nii.gz

if [ ! -f $nii_midEig_reg_norm ]; then
	$1 python normalize.py --input $nii_fa_reg --mask $nii_mask_reg --output $nii_fa_reg_norm --flip 1
	$1 python normalize.py --input $nii_trace_reg --mask $nii_mask_reg --output $nii_trace_reg_norm --flip 1
	$1 python normalize.py --input $nii_minEig_reg --mask $nii_mask_reg --output $nii_minEig_reg_norm --flip 1
	$1 python normalize.py --input $nii_midEig_reg --mask $nii_mask_reg --output $nii_midEig_reg_norm --flip 1
fi

# 8) 在 atlas 空间运行 2.5D 多视角 DDSurfer 推理。
#    这一步会生成核心输出：
#    - *-DDSurfer-wmparc-Reg.mgz（atlas 空间分割）
mgz_wmparc_reg=$outputdir/$subID-DDSurfer-wmparc-Reg.mgz
if [ ! -f $mgz_wmparc_reg ]; then
	$1 python DDSurfer_Pred.py --in_dir $outputdir --out_dir $outputdir --weights_dir ./weights/
fi

nii_wmparc=$outputdir/$subID-DDSurfer-wmparc.nii.gz
if [ ! -f $nii_wmparc ]; then
  # 9) 将分割结果变换回被试原始/mask 空间，供后续使用。
  #    输出：
  #    - *-DDSurfer-wmparc.nii.gz（原空间分割，更方便和原始数据对照）
  $1 $ResampleScalarVectorDWIVolume --Reference $mask --transformationFile $tfminv --interpolation nn $mgz_wmparc_reg $nii_wmparc
fi




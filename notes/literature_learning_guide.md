# DDParcel-CTAT 论文学习阅读指南

这份文档是学习用，不是论文写作综述。目标是按知识依赖顺序理解 DDParcel-CTAT 背后的技术链：diffusion MRI（扩散磁共振成像）和 DTI（diffusion tensor imaging，扩散张量成像）提供输入，FreeSurfer 提供 anatomical labels（解剖标签），U-Net/FastSurfer 提供 segmentation backbone（分割骨干网络），DDParcel 把任务迁移到 diffusion MRI，CTAT 再引入 token fusion（令牌融合）和 sparse attention（稀疏注意力）。

PDF 下载位置：`literature/pdfs/`。  
下载记录：`literature/download_manifest.json`。  
有些基础医学影像论文和 DDParcel 原论文属于出版社页面或 IEEE 页面，本项目只记录 DOI/页面链接，不绕过权限下载。

## 推荐阅读顺序

1. DTI 基础：Basser 1994 -> Le Bihan 2001 -> Mori 2002
2. FreeSurfer 标签体系：Fischl 2002 -> Fischl 2004 -> Desikan 2006
3. 医学图像分割基础：U-Net -> V-Net -> Litjens survey
4. DDParcel 架构前身：DenseNet -> Maxout -> QuickNAT -> FastSurfer
5. 本项目主体：DDParcel
6. CTAT 改动基础：Transformer -> ViT -> Swin -> TransUNet -> UNETR
7. 稀疏注意力基础：Sparsemax -> Entmax
8. 数据集背景：HCP -> CNP -> PPMI

---

## 01. Basser et al. 1994 - MR diffusion tensor spectroscopy and imaging

链接：https://doi.org/10.1016/S0006-3495(94)80775-1  
PDF 状态：出版社 DOI 页面，未自动下载。

### 为什么先读它
这篇是 DTI（diffusion tensor imaging，扩散张量成像）的源头之一。DDParcel 的输入不是普通 T1 MRI，而是 diffusion MRI 派生出的 scalar maps（标量图），例如 FA（fractional anisotropy，各向异性分数）、Trace（扩散张量迹）、MinEigenvalue（最小特征值）和 MidEigenvalue（中间特征值）。如果不理解 DTI，就很难理解为什么 DDParcel 会用这些模态。

### 一句话理解
这篇论文提出用一个 3x3 diffusion tensor（扩散张量）描述水分子在三维空间中不同方向的扩散强度，从而把 diffusion MRI 从“单方向强度图”变成“可计算方向结构的物理模型”。

### 核心概念
- diffusion（扩散）：水分子随机运动。
- anisotropy（各向异性）：不同方向扩散程度不同，白质纤维中尤其明显。
- diffusion tensor（扩散张量）：用矩阵表示三维扩散方向和强度。
- eigenvalue（特征值）：张量主方向上的扩散强度。
- FA（fractional anisotropy，各向异性分数）：衡量扩散方向性强弱。
- Trace（迹）：三个特征值之和，反映整体扩散强度。

### 方法拆解
传统 diffusion MRI 可以测某个方向上的扩散衰减，但难以完整描述三维扩散形状。Basser 这篇的关键是把每个 voxel（体素）里的扩散建模成 tensor（张量）。张量有 eigenvectors（特征向量）和 eigenvalues（特征值），前者描述主扩散方向，后者描述各方向扩散强度。这样就可以从原始扩散信号中计算出 FA、Trace 等后续模型可用的图像。

### 和 DDParcel-CTAT 的关系
DDParcel-CTAT 当前使用的四个输入模态都来自 DTI tensor 的派生量。项目中的 `FractionalAnisotropy`、`Trace`、`MinEigenvalue`、`MidEigenvalue` 不是随便选的图像通道，而是 diffusion tensor 的不同统计视角。

### 重点读哪里
重点读方法中 diffusion tensor 如何定义、如何由 MRI signal estimation（信号估计）得到，以及 eigenvalue/eigenvector 的解释。公式不必全部推完，但要理解“为什么一个 voxel 可以变成一个 tensor”。

### 自测问题
1. diffusion MRI 和 DTI 有什么区别？
2. 为什么白质区域通常有更高 anisotropy？
3. FA 和 Trace 分别表达什么信息？
4. DDParcel 为什么不用单一 FA 图，而要用多个 DTI scalar maps？

---

## 02. Le Bihan et al. 2001 - Diffusion tensor imaging: concepts and applications

链接：https://doi.org/10.1002/jmri.1076  
PDF 状态：出版社 DOI 页面，未自动下载。

### 为什么读它
如果 Basser 1994 是 DTI 的数学起点，这篇更像概念综述。它帮助把 tensor、anisotropy、fiber orientation（纤维方向）和临床/神经影像应用连起来。

### 一句话理解
这篇解释了 DTI 如何把水分子扩散方向变成可解释的脑组织结构信息，尤其适合学习 FA、eigenvalue、white matter integrity（白质完整性）这些概念。

### 核心概念
- ADC（apparent diffusion coefficient，表观扩散系数）：扩散强度的基础量。
- white matter tract（白质束）：神经纤维组成的结构路径。
- directional diffusivity（方向性扩散）：不同方向的扩散差异。
- scalar map（标量图）：从张量计算出的单通道图像，如 FA、MD、Trace。

### 方法拆解
这篇不只是推公式，而是解释 DTI 的物理含义。一个 tensor 可以看成椭球体：球越细长，方向性越强；越接近球形，扩散越均匀。FA 反映椭球“细长程度”，Trace 或 mean diffusivity 反映总体扩散强弱。这些量让深度模型可以从 diffusion MRI 中学习解剖结构。

### 和 DDParcel-CTAT 的关系
DDParcel 输入的四个模态可以理解成四种 tissue contrast（组织对比）。CTAT 做 modality competition（模态竞争）时，本质是在学习不同 DTI scalar maps 在不同脑区、不同空间位置上哪个更有用。

### 重点读哪里
重点看 DTI scalar quantities（标量量）、anisotropy interpretation（各向异性解释）和应用部分。读完应能解释每个 DDParcel 输入模态的生物物理含义。

### 自测问题
1. FA 高说明什么，FA 低一定说明病变吗？
2. Trace 和 mean diffusivity 有什么关系？
3. DTI scalar map 为什么可以作为 segmentation network（分割网络）的输入？

---

## 03. Mori & van Zijl 2002 - Fiber tracking: principles and strategies

链接：https://doi.org/10.1002/nbm.781  
PDF 状态：出版社 DOI 页面，未自动下载。

### 为什么读它
DDParcel 不是 tractography（纤维追踪）项目，但 diffusion MRI 的重要价值来自白质方向结构。读这篇能理解 diffusion signal 为什么对脑区边界和白质解剖有价值。

### 一句话理解
这篇解释如何根据 DTI 的主扩散方向追踪 white matter fibers（白质纤维），说明 diffusion MRI 能提供结构连接和解剖方向信息。

### 核心概念
- tractography（纤维追踪）：根据扩散方向重建白质路径。
- principal eigenvector（主特征向量）：最大扩散方向。
- streamline（流线）：沿主方向连续追踪得到的路径。
- crossing fibers（交叉纤维）：多个纤维方向混在一个 voxel 中的难点。

### 方法拆解
基本思路是：每个 voxel 有一个主方向，算法从 seed point（种子点）出发，沿相邻 voxel 的主方向一步步前进，形成 fiber tract（纤维束）。困难在于噪声、低 FA 区域、交叉纤维和 partial volume effect（部分容积效应）。

### 和 DDParcel-CTAT 的关系
DDParcel 不直接追踪纤维，但它利用 diffusion-derived maps（扩散派生图）作为 anatomical parcellation（解剖分区）的输入。理解 tractography 能帮助你理解为什么 diffusion MRI 不只是“另一种灰度图”，而是携带方向性解剖信息。

### 重点读哪里
重点看 principles（基本原理）和 limitations（局限）。不需要深入实现 tractography，但要理解 DTI 信息为什么有助于白质和深部结构识别。

### 自测问题
1. tractography 为什么依赖 principal eigenvector？
2. crossing fibers 为什么是 DTI 的难点？
3. DDParcel 用 DTI scalar maps 而不是 fiber tract，本质差别是什么？

---

## 04. Fischl et al. 2002 - Whole brain segmentation

链接：https://doi.org/10.1016/S0896-6273(02)00569-X  
PDF 状态：出版社 DOI 页面，未自动下载。

### 为什么读它
这篇是 FreeSurfer 体素级 subcortical segmentation（皮层下结构分割）的基础。DDParcel 输出遵循 FreeSurfer 风格标签，因此必须理解这个标签体系从哪里来。

### 一句话理解
这篇提出自动给 MRI 体数据中的脑结构打 anatomical labels（解剖标签），为后续 aparc+aseg 标签体系奠定基础。

### 核心概念
- whole brain segmentation（全脑分割）：给整脑体素分配结构标签。
- atlas prior（图谱先验）：利用已知脑结构空间分布辅助分割。
- intensity model（强度模型）：不同组织/结构在 MRI 上的强度分布。
- aseg（automatic segmentation）：FreeSurfer 体积分割输出。

### 方法拆解
传统手工标注非常慢且主观。FreeSurfer 用 atlas prior、MRI intensity 和空间关系做自动标注。核心思想是：一个 voxel 属于某个结构，不只由它的灰度决定，还由它在脑中的位置和邻近结构决定。

### 和 DDParcel-CTAT 的关系
DDParcel 的 ground truth labels（训练标签）来自 FreeSurfer/FreeSurfer 风格分区。代码里的 `map_label2aparc_aseg`、`map_aparc_aseg2label` 就是在网络内部连续类别和 FreeSurfer 标签 ID 之间转换。

### 重点读哪里
重点读 segmentation framework（分割框架）和 atlas/intensity prior 如何结合。不要只看结果，要理解标签体系背后的假设。

### 自测问题
1. 为什么脑结构分割不能只靠 intensity？
2. atlas prior 在分割里起什么作用？
3. DDParcel 内部类别索引和 FreeSurfer label ID 为什么需要映射？

---

## 05. Fischl et al. 2004 - Automatically parcellating the human cerebral cortex

链接：https://doi.org/10.1093/cercor/bhg087  
PDF 状态：出版社 DOI 页面，未自动下载。

### 为什么读它
DDParcel 的任务是 brain parcellation（脑区分区），不只是 tissue segmentation（组织分割）。这篇是 FreeSurfer cortical parcellation（皮层分区）的核心来源。

### 一句话理解
这篇把大脑皮层按 gyral/sulcal anatomy（脑回/脑沟解剖）自动划分成不同 cortical regions（皮层区域）。

### 核心概念
- cortical parcellation（皮层分区）：把皮层分成有解剖意义的区域。
- sulcus/gyrus（脑沟/脑回）：皮层折叠结构，是皮层分区的重要依据。
- surface-based analysis（基于表面的分析）：在皮层表面而不是体素空间中分析。
- anatomical label（解剖标签）：具有神经解剖意义的区域 ID。

### 方法拆解
皮层不是简单三维块状结构，而是一张折叠表面。FreeSurfer 会重建 cortical surface（皮层表面），再根据折叠模式和 atlas 信息给区域命名。这个任务比简单灰白质分割更细，因为需要区分多个相邻皮层脑区。

### 和 DDParcel-CTAT 的关系
DDParcel 最终预测的是类似 FreeSurfer `aparc+aseg` 的区域标签。理解这篇有助于明白为什么输出类别多、标签 ID 看起来不连续，以及为什么左右半球和皮层标签需要特殊处理。

### 重点读哪里
重点看 parcellation 的定义、surface registration（表面配准）和 atlas labeling（图谱标注）。对 DDParcel 来说，要理解输出标签的语义，而不是只把它当作 102 类分类。

### 自测问题
1. segmentation 和 parcellation 的区别是什么？
2. 为什么 cortical parcellation 常用 surface-based 方法？
3. DDParcel 从 dMRI 预测 FreeSurfer 风格标签有什么挑战？

---

## 06. Desikan et al. 2006 - Automated labeling system for subdividing the cortex

链接：https://doi.org/10.1016/j.neuroimage.2006.01.021  
PDF 状态：出版社 DOI 页面，未自动下载。

### 为什么读它
这篇定义了常见的 Desikan-Killiany atlas（DK 图谱）。很多 FreeSurfer cortical labels（皮层标签）来自这个体系。

### 一句话理解
这篇提供了一个可重复的 cortical labeling system（皮层标注系统），把人脑皮层划分成一组标准解剖区域。

### 核心概念
- atlas（图谱）：标准化区域划分。
- ROI（region of interest，感兴趣区域）：一个可分析的脑区。
- gyral-based labeling（基于脑回的标注）：根据解剖折叠结构分区。
- inter-rater reliability（一致性）：不同标注者能否得到相似结果。

### 方法拆解
论文的重点不是提出复杂神经网络，而是定义和验证一个可复用的标签系统。它回答的问题是：怎样把皮层区域划分得既符合解剖，又能让不同研究者复现。

### 和 DDParcel-CTAT 的关系
DDParcel 预测的 cortical classes（皮层类别）需要有稳定语义。DK atlas 提供了许多标签名称和区域边界的背景。

### 重点读哪里
重点看 atlas definition（图谱定义）和 reproducibility（可重复性）部分。读完应该能理解“标签不是随便编号，而是有解剖定义”。

### 自测问题
1. atlas 和训练标签有什么关系？
2. 为什么医学分割模型需要关心标签体系的可靠性？
3. DDParcel 的 label remapping 为什么是方法的一部分？

---

## 07. Ronneberger et al. 2015 - U-Net

链接：https://arxiv.org/abs/1505.04597  
PDF：`literature/pdfs/07_unet_2015.pdf`

### 为什么读它
U-Net 是医学图像分割的基础架构。FastSurfer、QuickNAT、DDParcel 的很多结构都可以看成 U-Net 思想的变体。

### 一句话理解
U-Net 用 encoder-decoder（编码器-解码器）和 skip connection（跳跃连接）同时获得语义信息和精细空间定位。

### 核心概念
- encoder（编码器）：逐步下采样，提取高层语义。
- decoder（解码器）：逐步上采样，恢复分辨率。
- skip connection（跳跃连接）：把浅层细节传到解码器。
- semantic segmentation（语义分割）：给每个像素/体素分类。

### 方法拆解
U-Net 左边收缩路径提取上下文，右边扩张路径恢复空间尺寸。普通分类网络会丢失空间细节，而 U-Net 通过 skip connection 把高分辨率浅层特征送到 decoder，使边界更清楚。医学影像数据通常少，U-Net 也强调 data augmentation（数据增强）。

### 和 DDParcel-CTAT 的关系
DDParcel 的核心网络继承 U-Net 式结构，只是输入变为 2.5D thick slices（厚切片），并加入 competitive dense block（竞争式密集块）和多模态融合。

### 重点读哪里
重点看 Figure 1 的结构图，理解 encoder/decoder/skip connection。实验部分看它如何在少量医学标注数据下工作。

### 自测问题
1. U-Net 为什么适合医学图像分割？
2. skip connection 解决了什么问题？
3. DDParcel 和 U-Net 相比，任务和输入发生了哪些变化？

---

## 08. Milletari et al. 2016 - V-Net

链接：https://arxiv.org/abs/1606.04797  
PDF：`literature/pdfs/08_vnet_2016.pdf`

### 为什么读它
V-Net 是 3D medical segmentation（医学三维分割）的经典工作，也常和 Dice loss（Dice 损失）一起被引用。

### 一句话理解
V-Net 把 U-Net 思想扩展到 3D volume（体数据），并使用 Dice-based objective（基于 Dice 的目标函数）直接优化分割重叠。

### 核心概念
- 3D convolution（三维卷积）：在体数据中同时利用三个空间维度。
- Dice coefficient（Dice 系数）：衡量预测分割和真实标签重叠程度。
- class imbalance（类别不平衡）：医学分割中小结构体素很少。
- volumetric segmentation（体积分割）：对 3D volume 中每个 voxel 分类。

### 方法拆解
U-Net 是 2D 结构，V-Net 直接处理 3D patch。它强调 Dice loss，因为医学分割中背景占比大，cross entropy 容易被大类主导。Dice loss 关注预测和标签的重叠比例，更适合小结构分割。

### 和 DDParcel-CTAT 的关系
DDParcel 采用 2.5D 多视角而不是完整 3D，原因是计算成本和训练稳定性。CTAT 的训练里也使用 Dice/CE 组合损失，这与医学分割的类别不平衡问题直接相关。

### 重点读哪里
重点看 Dice loss 的定义和为什么要用 3D context（3D 上下文）。对本项目来说，要理解“为什么分割 loss 不能只看分类准确率”。

### 自测问题
1. Dice coefficient 和 voxel accuracy 有什么区别？
2. 为什么小脑区分割会有 class imbalance？
3. DDParcel 为什么选择 2.5D 多视角而不是纯 3D V-Net？

---

## 09. Litjens et al. 2017 - Deep learning in medical image analysis survey

链接：https://arxiv.org/abs/1702.05747  
PDF：`literature/pdfs/09_litjens_survey_2017.pdf`

### 为什么读它
这是一篇医学影像深度学习综述，适合把 segmentation、classification、registration、detection 等任务放进大图景里。

### 一句话理解
这篇总结了深度学习如何改变医学图像分析，并解释不同任务、数据规模、标注和评估方式的常见问题。

### 核心概念
- medical image analysis（医学图像分析）：医学图像中的分类、分割、检测、配准等任务。
- data scarcity（数据稀缺）：医学标注数据少。
- annotation cost（标注成本）：专家标注昂贵。
- evaluation metric（评价指标）：Dice、Hausdorff distance 等。

### 方法拆解
综述不是一个方法，而是地图。它帮助你理解 DDParcel 属于 segmentation/parcellation 方向，面临医学影像共有问题：数据少、类别不平衡、跨数据集泛化难、标注依赖专家。

### 和 DDParcel-CTAT 的关系
这篇提供宏观背景。DDParcel-CTAT 的很多工程选择，例如预训练 backbone、2.5D 输入、数据增强、Dice loss，都能在医学影像深度学习常见挑战中找到原因。

### 重点读哪里
重点读 segmentation 相关章节、data augmentation、training with limited data 和 evaluation 部分。

### 自测问题
1. 医学图像分割和自然图像分割有什么不同？
2. 为什么医学影像模型容易出现泛化问题？
3. DDParcel-CTAT 的哪些设计是在应对数据稀缺？

---

## 10. Huang et al. 2017 - DenseNet

链接：https://arxiv.org/abs/1608.06993  
PDF：`literature/pdfs/10_densenet_2017.pdf`

### 为什么读它
FastSurfer 使用 competitive dense block（竞争式密集块），其背景来自 DenseNet 的 dense connectivity（密集连接）。

### 一句话理解
DenseNet 让每一层都接收前面所有层的特征，从而增强 feature reuse（特征复用）和 gradient flow（梯度流动）。

### 核心概念
- dense connection（密集连接）：每层连接到所有之前层。
- feature reuse（特征复用）：后层直接使用早期特征。
- growth rate（增长率）：每层新增特征通道数。
- gradient flow（梯度流）：梯度更容易传到浅层。

### 方法拆解
普通 CNN 每层只接上一层，DenseNet 把前面所有层 concat（拼接）起来作为输入。这样网络可以复用边缘、纹理、局部结构等低层特征，也缓解梯度消失。

### 和 DDParcel-CTAT 的关系
FastSurferCNN 的 block 名称和思想都受 DenseNet 影响。但 FastSurfer/DDParcel 不是简单 concat，而是用 maxout competition（最大值竞争）控制通道数量。

### 重点读哪里
重点看 dense block 结构图和 feature reuse 的解释。读完再看 FastSurfer 时，才能理解 competitive dense block 改了什么。

### 自测问题
1. DenseNet 和 ResNet 的连接方式有什么不同？
2. dense connection 为什么有利于医学分割？
3. FastSurfer 为什么不直接照搬 concat？

---

## 11. Goodfellow et al. 2013 - Maxout Networks

链接：https://arxiv.org/abs/1302.4389  
PDF：`literature/pdfs/11_maxout_2013.pdf`

### 为什么读它
FastSurfer 的 competitive dense block 使用 maxout（最大值输出）思想：多个候选特征竞争，只保留最大响应。

### 一句话理解
Maxout 用一组线性/特征响应的最大值作为输出，让网络学习更灵活的激活函数。

### 核心概念
- maxout（最大输出）：从多个候选响应中取最大值。
- competition（竞争）：不同特征通道之间通过最大值选择胜出者。
- activation function（激活函数）：决定神经元非线性响应。
- model capacity（模型容量）：模型表达复杂函数的能力。

### 方法拆解
ReLU 是固定形状的非线性，maxout 让模型从多个响应中选择最大值，相当于学习一个 piecewise linear function（分段线性函数）。在 FastSurfer 中，maxout 也有降维和选择特征的作用。

### 和 DDParcel-CTAT 的关系
DDParcel 的 fusion network 继承 FastSurfer 的竞争机制。CTAT 的 modality competition（模态竞争）可以看作把这种“竞争选择”从卷积特征扩展到 token/modality 层面。

### 重点读哪里
重点理解 maxout 单元定义和为什么它表示能力强。不必深究 dropout 细节。

### 自测问题
1. maxout 和 ReLU 的区别是什么？
2. competition 机制在 FastSurfer 中有什么工程好处？
3. CTAT 的 sparse attention 和 maxout competition 有什么思想相似点？

---

## 12. Roy et al. 2019 - QuickNAT

链接：https://arxiv.org/abs/1801.04161  
PDF：`literature/pdfs/12_quicknat_2019.pdf`

### 为什么读它
QuickNAT 是 FastSurfer 的直接前身。它展示了用三个 2D networks（二维网络）处理三个正交视角来做快速脑分割。

### 一句话理解
QuickNAT 用 2D fully convolutional networks（全卷积网络）在 axial/coronal/sagittal 三个视角上预测脑结构标签，再融合结果。

### 核心概念
- 2.5D segmentation（2.5D 分割）：用相邻切片提供 3D 上下文，但仍用 2D 网络。
- multi-view inference（多视角推理）：从多个正交方向预测并融合。
- whole brain segmentation（全脑分割）：分割多个脑结构。
- slice-wise prediction（逐切片预测）：每次预测一个切片。

### 方法拆解
完整 3D 网络计算量大，2D 网络缺少三维上下文。QuickNAT 折中：每个视角训练一个 2D 分割模型，利用相邻切片作为输入，最后融合三个方向的预测。这样速度快，也能获得一定 3D 信息。

### 和 DDParcel-CTAT 的关系
DDParcel 也使用 axial/coronal/sagittal 三视角推理。你看到的 `DDSurfer_Pred.py` 中 axial/coronal/sagittal 加载，就是这种路线的延续。

### 重点读哪里
重点看三视角设计和快速推理动机。理解后再看 DDParcel 的 three-view fusion 会更自然。

### 自测问题
1. 2D、2.5D、3D segmentation 各有什么优缺点？
2. 为什么三视角融合可以补充单视角不足？
3. DDParcel 从 QuickNAT 继承了哪些流程？

---

## 13. Henschel et al. 2020 - FastSurfer

链接：https://arxiv.org/abs/1910.03866  
PDF：`literature/pdfs/13_fastsurfer_2020.pdf`

### 为什么读它
FastSurfer 是 DDParcel backbone（骨干网络）的关键来源。DDParcel 不是从零发明网络，而是在 FastSurferCNN 的基础上扩展到 diffusion MRI 和多模态融合。

### 一句话理解
FastSurfer 用 competitive dense blocks 构建快速准确的脑分割 pipeline，大幅加速 FreeSurfer 风格分割。

### 核心概念
- FastSurferCNN：FastSurfer 的分割 CNN。
- competitive dense block（竞争式密集块）：dense connection + maxout competition。
- aparc+aseg：FreeSurfer 的皮层加皮层下标签输出。
- inference speed（推理速度）：相对传统 FreeSurfer 的重要优势。

### 方法拆解
FastSurfer 保留 U-Net 编码器-解码器结构，但用 competitive dense blocks 替换普通卷积块。相比 DenseNet 的 concat，它通过 maxout 控制通道数，减少内存和计算。它仍然用多视角 2.5D 策略来分割全脑。

### 和 DDParcel-CTAT 的关系
DDParcel 的 `FastSurferCNN_Fuse_Unet_v3_extended` 等模型来自 FastSurferCNN 设计。CTAT 项目则是在 DDParcel 的多模态输入之上，用 Transformer-style token fusion 改进模态选择。

### 重点读哪里
重点读 network architecture、competitive dense block 和 evaluation against FreeSurfer。要理解“为什么它快”和“为什么它能输出 FreeSurfer 风格标签”。

### 自测问题
1. FastSurfer 相比 FreeSurfer 快在哪里？
2. competitive dense block 和 DenseNet block 有什么差别？
3. DDParcel 为什么选择 FastSurferCNN 作为 backbone？

---

## 14. Zhang et al. 2024 - DDParcel

链接：https://doi.org/10.1109/TMI.2023.3331691  
代码：https://github.com/zhangfanmark/DDParcel  
PDF 状态：IEEE 页面，未自动下载。

### 为什么读它
这是本项目的直接原论文。DDParcel-CTAT 的大部分数据预处理、标签映射、三视角推理和多模态 diffusion MRI 输入都继承自它。

### 一句话理解
DDParcel 把 FreeSurfer 风格 anatomical brain parcellation（解剖脑区分区）从 T1 MRI 转移到 diffusion MRI，并使用多个 DTI scalar maps 进行深度学习分割。

### 核心概念
- brain parcellation（脑区分区）：把脑划分为多个解剖区域。
- diffusion MRI（扩散磁共振）：提供组织微结构和白质信息。
- multi-modal fusion（多模态融合）：融合 FA、Trace、eigenvalue 等输入。
- three-view inference（三视角推理）：axial/coronal/sagittal 分别预测再融合。
- label remapping（标签重映射）：网络内部类别和 FreeSurfer 标签之间转换。

### 方法拆解
DDParcel 的输入是四个预处理后的 DTI scalar maps。每个模态可以有自己的 backbone，融合网络在不同 encoder/decoder 层利用多个模态特征。推理时对三个方向分别运行模型，得到 probability maps（概率图），再融合、argmax 和映射回 FreeSurfer label IDs。

### 和 DDParcel-CTAT 的关系
CTAT 是在 DDParcel 基础上的方法扩展。现有项目保留 DDParcel 的数据、标签和推理背景，但增加 Transformer-like competitive token attention（竞争式令牌注意力），希望更精细地学习模态之间的选择关系。

### 重点读哪里
重点读数据输入、网络融合结构、三视角推理、实验数据集和评价指标。尤其要对照代码看 `DDSurfer_Pred.py`、`models/networks.py` 和 `data_loader/load_neuroimaging_data.py`。

### 自测问题
1. DDParcel 为什么要从 diffusion MRI 做 parcellation？
2. 四个 DTI scalar maps 分别提供什么信息？
3. DDParcel 的多模态融合和 CTAT 的 token fusion 有什么区别？
4. 为什么 DDParcel 的 demo 不能等同于完整训练数据？

---

## 15. Vaswani et al. 2017 - Attention Is All You Need

链接：https://arxiv.org/abs/1706.03762  
PDF：`literature/pdfs/15_attention_2017.pdf`

### 为什么读它
CTAT 使用 attention/token 的思想，源头是 Transformer。即使本项目不是 NLP，也需要理解 query/key/value 和 attention weights。

### 一句话理解
Transformer 用 self-attention（自注意力）让每个 token 根据与其他 token 的关系动态聚合信息，而不依赖卷积或循环。

### 核心概念
- token（令牌）：模型处理的基本单元。
- self-attention（自注意力）：同一序列内部 token 互相建模。
- query/key/value：计算注意力的三组表示。
- multi-head attention（多头注意力）：并行学习多种关系。
- positional encoding（位置编码）：补充顺序/空间位置信息。

### 方法拆解
每个 token 生成 query、key、value。query 和 key 做相似度，softmax 后得到 attention weights（注意力权重），再加权 value。这样模型可以根据内容动态选择信息来源。multi-head attention 让模型同时关注不同关系。

### 和 DDParcel-CTAT 的关系
CTAT 把医学图像特征切成 tokens，并在模态之间做 attention/competition。理解 Transformer 是理解 CTAT 的第一步。

### 重点读哪里
重点看 scaled dot-product attention 和 multi-head attention。NLP 的 encoder/decoder 细节可以略读。

### 自测问题
1. query/key/value 分别做什么？
2. attention 和卷积的主要区别是什么？
3. 医学图像 token 为什么需要 positional encoding？

---

## 16. Dosovitskiy et al. 2020 - Vision Transformer

链接：https://arxiv.org/abs/2010.11929  
PDF：`literature/pdfs/16_vit_2020.pdf`

### 为什么读它
ViT 把 Transformer 从文本迁移到图像。CTAT 的 tokenization（令牌化）思想和 ViT 直接相关。

### 一句话理解
ViT 把图像切成 patches（图像块），把每个 patch 当作 token，再用 Transformer 做视觉识别。

### 核心概念
- patch embedding（图像块嵌入）：把图像块转成 token 向量。
- class token（分类令牌）：用于图像级分类的聚合 token。
- positional embedding（位置嵌入）：告诉模型 patch 的空间位置。
- global attention（全局注意力）：所有 patch 两两交互。

### 方法拆解
图像被切成固定大小 patch，例如 16x16。每个 patch flatten 后通过 linear projection 变成 token。Transformer 处理 token 序列，最后用 class token 做分类。ViT 的关键启发是：图像也可以像句子一样被 token 化。

### 和 DDParcel-CTAT 的关系
CTAT 不是做图像分类，而是做 segmentation，但它同样需要把空间特征变成 tokens。区别是 CTAT 最终还要恢复 dense prediction（密集预测），不能只输出一个 class token。

### 重点读哪里
重点看 patch embedding 和 positional embedding。实验部分可快速看，重点理解大数据预训练对 ViT 的重要性。

### 自测问题
1. patch token 和 CNN feature map 有什么关系？
2. ViT 为什么需要 positional embedding？
3. segmentation 使用 token 后，如何恢复空间布局？

---

## 17. Liu et al. 2021 - Swin Transformer

链接：https://arxiv.org/abs/2103.14030  
PDF：`literature/pdfs/17_swin_2021.pdf`

### 为什么读它
Swin 引入 window attention（窗口注意力）和 shifted window（移位窗口），适合高分辨率视觉任务。CTAT 中 window/token 设计与这个思想更接近。

### 一句话理解
Swin Transformer 通过局部窗口注意力降低计算量，并用窗口移位实现跨窗口信息交互。

### 核心概念
- window attention（窗口注意力）：只在局部窗口内计算 attention。
- shifted window（移位窗口）：下一层移动窗口边界，让不同窗口互相通信。
- hierarchical representation（层级表示）：像 CNN 一样逐层降采样。
- computational complexity（计算复杂度）：全局 attention 在大图上代价高。

### 方法拆解
全局 self-attention 对高分辨率图像代价太大。Swin 把图像划成窗口，每个窗口内部做 attention。为了避免窗口之间完全隔离，下一层把窗口平移，让边界处 token 能跨窗口交互。

### 和 DDParcel-CTAT 的关系
医学分割要处理 256x256 或 3D volume，直接全局 attention 很贵。CTAT 中 window-based token design 可以借鉴 Swin 的局部建模和计算控制。

### 重点读哪里
重点看 W-MSA、SW-MSA 和 hierarchical design。读完要能解释为什么 window attention 比 global attention 更适合大图。

### 自测问题
1. Swin 为什么不用全局 attention？
2. shifted window 解决了什么问题？
3. CTAT 如果做 token fusion，为什么要关心计算复杂度？

---

## 18. Chen et al. 2021 - TransUNet

链接：https://arxiv.org/abs/2102.04306  
PDF：`literature/pdfs/18_transunet_2021.pdf`

### 为什么读它
TransUNet 是医学图像分割中 CNN + Transformer 的经典路线。它连接 U-Net 和 Transformer，是理解 CTAT 的重要桥梁。

### 一句话理解
TransUNet 用 CNN 提取局部特征，用 Transformer 建模长程依赖，再用 U-Net decoder 恢复分割图。

### 核心概念
- hybrid CNN-Transformer（混合 CNN-Transformer）：CNN 负责局部纹理，Transformer 负责全局关系。
- long-range dependency（长程依赖）：远距离区域之间的关系。
- decoder upsampling（解码器上采样）：把 token/特征恢复到像素级输出。
- medical segmentation（医学分割）：对器官/结构做 dense prediction。

### 方法拆解
TransUNet 不直接用原图 patch，而是先用 CNN 得到 feature map，再把 feature map token 化给 Transformer。Transformer 输出再接 U-Net 式 decoder。这样保留 CNN 的局部归纳偏置，也获得 Transformer 的全局建模能力。

### 和 DDParcel-CTAT 的关系
CTAT 也是在 CNN segmentation backbone 上加入 token attention。区别是 CTAT 的重点不是单模态全局建模，而是 modality-competitive token fusion（模态竞争式令牌融合）。

### 重点读哪里
重点看 hybrid architecture 和 decoder 如何恢复空间输出。理解后再看 CTAT 的 skip tokens 和 spatial reconstruction。

### 自测问题
1. TransUNet 为什么不完全抛弃 CNN？
2. Transformer 在医学分割里补充了什么？
3. CTAT 和 TransUNet 的核心关注点有什么不同？

---

## 19. Hatamizadeh et al. 2021 - UNETR

链接：https://arxiv.org/abs/2103.10504  
PDF：`literature/pdfs/19_unetr_2021.pdf`

### 为什么读它
UNETR 是 3D 医学图像分割中的 Transformer encoder 代表。它帮助理解 token-based 3D segmentation。

### 一句话理解
UNETR 把 3D volume 切成 patches，用 Transformer encoder 学习全局表示，再用 CNN decoder 输出三维分割。

### 核心概念
- 3D patch embedding（三维图像块嵌入）：把体数据分块成 tokens。
- transformer encoder（Transformer 编码器）：学习 token 间关系。
- skip connection from transformer layers（来自 Transformer 层的跳跃连接）：把不同深度表示接入 decoder。
- volumetric medical segmentation（三维医学分割）：体素级结构预测。

### 方法拆解
UNETR 直接把 3D volume token 化，Transformer 处理所有 3D patches。decoder 从不同 Transformer 层取特征，逐步恢复空间分辨率。这说明 Transformer 不只是分类器，也可以服务于 dense 3D prediction。

### 和 DDParcel-CTAT 的关系
DDParcel-CTAT 当前主要是 2.5D，但它面临的空间恢复、skip feature 和 token layout 问题与 UNETR 相通。

### 重点读哪里
重点看 3D patch embedding 和 decoder skip connections。实验细节可略读。

### 自测问题
1. UNETR 如何把 3D volume 变成 token sequence？
2. Transformer 输出如何恢复为 3D segmentation？
3. CTAT 的 token-to-spatial 测试为什么重要？

---

## 20. Martins & Astudillo 2016 - Sparsemax

链接：https://arxiv.org/abs/1602.02068  
PDF：`literature/pdfs/20_sparsemax_2016.pdf`

### 为什么读它
CTAT demo 中比较了 softmax 和 sparsemax。理解 sparsemax 是理解“真正竞争”的关键。

### 一句话理解
Sparsemax 把分数映射成概率分布，但允许很多概率精确为 0，从而产生稀疏选择。

### 核心概念
- softmax（软最大）：所有位置概率通常都大于 0。
- sparsemax（稀疏最大）：部分位置概率可以等于 0。
- probability simplex（概率单纯形）：概率非负且和为 1 的空间。
- sparse attention（稀疏注意力）：只激活少数 token。

### 方法拆解
softmax 会给每个候选项一点概率，即使很小。sparsemax 相当于把分数投影到 probability simplex 上，低分候选会被压到 0。这样模型不是“所有模态都稍微用一点”，而是可以真正排除某些模态/token。

### 和 DDParcel-CTAT 的关系
CTAT 的核心卖点之一是 modality competition。Sparsemax 提供了数学工具，让模态选择更稀疏、更可解释。

### 重点读哪里
重点看 sparsemax 的定义、和 softmax 的差别，以及稀疏输出的例子。公式可以先理解几何直觉。

### 自测问题
1. softmax 为什么通常不产生精确 0？
2. sparsemax 的稀疏性对模态选择有什么意义？
3. CTAT demo 中 sparsemax near-zero 比例高说明什么？

---

## 21. Peters et al. 2019 - Sparse Sequence-to-Sequence Models / Entmax

链接：https://arxiv.org/abs/1905.05702  
PDF：`literature/pdfs/21_entmax_2019.pdf`

### 为什么读它
CTAT 使用 alpha-entmax annealing（alpha-entmax 退火），从 dense attention 逐渐过渡到 sparse attention。

### 一句话理解
Entmax 是 softmax 和 sparsemax 之间的一族变换，可以通过 alpha 控制注意力分布的稀疏程度。

### 核心概念
- entmax：可调稀疏概率映射。
- alpha（α）：控制从 softmax-like 到 sparsemax-like 的稀疏程度。
- annealing（退火）：训练过程中逐步改变超参数。
- sparse selection（稀疏选择）：只让少数候选被激活。

### 方法拆解
softmax 太 dense，sparsemax 可能太硬。Entmax 用 alpha 控制中间状态。alpha 接近 1 时像 softmax，alpha 接近 2 时像 sparsemax。训练初期可以让模型探索更多模态，后期逐步稀疏化，提高竞争性和解释性。

### 和 DDParcel-CTAT 的关系
项目 demo 已显示 active tokens 随 alpha 从 1 到 2 明显减少。这个机制可以解释 CTAT 为什么用 alpha schedule，而不是一开始就强制 sparsemax。

### 重点读哪里
重点看 entmax family、alpha 的意义和 attention sparsity。NLP seq2seq 实验不是本项目重点。

### 自测问题
1. entmax 和 sparsemax 的关系是什么？
2. 为什么训练初期不一定希望 attention 太稀疏？
3. CTAT 中 alpha schedule 解决了什么训练问题？

---

## 22. Human Connectome Project (HCP)

链接：https://www.humanconnectome.org/  
PDF 状态：项目/数据集参考，不作为单篇必读 PDF 自动下载。

### 为什么读它
HCP 是高质量人脑 MRI/dMRI 数据的重要来源。DDParcel demo 的 subject 命名和数据风格都与 HCP 相关。

### 一句话理解
HCP 提供大规模、高质量、多模态人脑影像数据，用于研究脑结构和连接。

### 核心概念
- multi-modal neuroimaging（多模态神经影像）：T1、T2、dMRI、fMRI 等。
- connectome（连接组）：脑区之间结构或功能连接图谱。
- data preprocessing（数据预处理）：配准、校正、标准化等。
- subject-level split（被试级划分）：训练/验证/测试按被试划分。

### 和 DDParcel-CTAT 的关系
DDParcel 论文使用 HCP 类数据训练/验证。我们当前 demo subject 只能做工程 smoke test，不能替代真正多被试训练。

### 自测问题
1. 为什么 HCP 对 dMRI 方法很重要？
2. 为什么不能按 slice 切分训练/测试？
3. demo subject 和正式训练集有什么区别？

---

## 23. CNP / OpenNeuro ds000030

链接：https://openneuro.org/datasets/ds000030  
PDF 状态：数据集参考，不作为单篇必读 PDF 自动下载。

### 为什么读它
CNP 是 DDParcel 相关数据来源之一，用于跨数据集验证时可能出现。

### 一句话理解
CNP 是公开神经精神表型数据集，包含多模态 neuroimaging 和行为/临床信息。

### 核心概念
- OpenNeuro：开放神经影像数据平台。
- phenotype（表型）：行为、临床、认知等个体特征。
- cross-dataset evaluation（跨数据集评估）：在不同来源数据上测试泛化。
- domain shift（域偏移）：不同扫描协议/人群导致数据分布变化。

### 和 DDParcel-CTAT 的关系
如果后续做论文级实验，CNP 可用于泛化评估。它提醒我们不能只在一个数据来源上证明方法有效。

### 自测问题
1. 跨数据集评估为什么比单数据集更难？
2. domain shift 会怎样影响 brain parcellation？
3. CTAT 的模态竞争是否可能改善跨域泛化？

---

## 24. PPMI

链接：https://www.ppmi-info.org/  
PDF 状态：数据集参考，不作为单篇必读 PDF 自动下载。

### 为什么读它
PPMI 是帕金森病相关的大规模纵向数据项目。DDParcel 论文提到这类数据时，通常用于疾病或跨人群验证。

### 一句话理解
PPMI 提供 Parkinson's disease（帕金森病）相关影像、临床和生物标志物数据，用于研究疾病进展。

### 核心概念
- longitudinal dataset（纵向数据集）：同一被试多个时间点。
- biomarker（生物标志物）：可用于疾病检测或进展跟踪的指标。
- disease cohort（疾病队列）：特定疾病人群数据。
- generalization（泛化）：模型在不同人群和疾病状态下是否稳定。

### 和 DDParcel-CTAT 的关系
如果 DDParcel-CTAT 要证明临床价值，不能只在健康 HCP 上表现好，还要考虑疾病队列、扫描差异和解剖变化。

### 自测问题
1. 疾病队列为什么会增加分割难度？
2. longitudinal data 对 parcellation 有什么额外要求？
3. PPMI 适合验证模型的哪类能力？

---

## 总结：这条知识线如何服务 DDParcel-CTAT

DTI 基础论文解释输入为什么是 FA、Trace 和 eigenvalue maps。FreeSurfer/DK atlas 论文解释输出标签为什么是 anatomical parcellation 而不是普通类别。U-Net、DenseNet、Maxout、QuickNAT 和 FastSurfer 解释 DDParcel 的 CNN backbone 和三视角 2.5D 推理。DDParcel 原论文把这些组件组合到 diffusion MRI brain parcellation。Transformer、ViT、Swin、TransUNet 和 UNETR 解释 CTAT 为什么可以引入 token-based fusion。Sparsemax 和 Entmax 解释 CTAT 为什么强调 modality competition 和 alpha annealing。HCP/CNP/PPMI 则定义后续真正实验需要面对的数据和泛化问题。

学习时不要急着读所有公式。第一遍目标是建立概念地图：输入是什么、输出是什么、网络为什么这样设计、每个设计解决哪个问题。第二遍再回到 DDParcel-CTAT 代码，把论文概念映射到文件和函数。

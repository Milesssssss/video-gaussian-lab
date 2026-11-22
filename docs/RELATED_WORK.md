# 相关工作与证据边界

核验日期：2026-11-22。范围：视频逐帧高斯初始化、外部深度校准、多视图融合和持久场景表示。

**本仓库已有 Phase 1 执行基础和合成测试，但没有下载或运行下列模型，也没有复现论文指标。** SHARP 与 Depth Pro 2 的公开模型许可限定非商业科学研究，并明确排除产品研发；本方案中的 Apple 后端是有许可前置条件的候选，不是默认可商用方案。执行条件和替代路线见 [THIRD_PARTY.md](THIRD_PARTY.md)，代码范围见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)。

## 1. 对本项目最有用的结论

已有工程原型把多帧 SHARP 高斯变换到共同世界坐标，再做深度尺度校准和体素融合；也已有项目把外部深度送入 SHARP 初始化。另有研究通过跨帧匹配、持久空间 token 或全局场景 token 直接学习场景融合。以下证据支持“路线值得开展受控实验”，不构成“SHARP + Depth Pro 2-L 已被验证能恢复完整一致场景”的结论。

本阶段采用固定实测位姿、冻结候选模型、逐帧全局尺度校准的设计，先分开测量初始化误差、几何不一致、冗余和渲染误差。局部深度形变、SHARP 内部深度注入、学习式全局特征融合均保留为后续研究；本阶段只按协议优化共享场景参数，不训练或微调网络。技术假设与实验门槛分别见 [TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md)、[PHASE1_SPEC.md](PHASE1_SPEC.md) 和 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)。

## 2. 固定的阅读版本

以下 GitHub 链接固定到完整 commit；论文固定到 arXiv 版本。固定源码版本不等于固定了权重：本项目尚无权重文件校验值。

| 工作 | 本次核验版本 | 已检查证据 | 本项目执行状态 |
|---|---|---|---|
| Apple SHARP | [aed6527499ef91cba3b54c18d49a870f25947190](https://github.com/apple-aiml-research/ml-sharp/tree/aed6527499ef91cba3b54c18d49a870f25947190) | 官方说明、预测器、代码及模型许可 | 未运行 |
| Apple Depth Pro 2-L | [aa389cbb0b74341664ed663a0966c859905cd6c0](https://github.com/apple-aiml-research/ml-depth-pro-2/tree/aa389cbb0b74341664ed663a0966c859905cd6c0)；[论文 v1](https://arxiv.org/abs/2611.26603v1) | 官方说明、推理接口、代码及模型许可 | 未运行 |
| ACCORD | [0a90e98ec793339c8a72ebf7e57540bb74e4b899](https://github.com/be2rlab/ACCORD/tree/0a90e98ec793339c8a72ebf7e57540bb74e4b899) | 官方说明、系统结构、许可 | 未运行 |
| camera-comfyUI | [a2f48da1a4a282eae08966c58ca243658a4f7e84](https://github.com/Alexankharin/camera-comfyUI/tree/a2f48da1a4a282eae08966c58ca243658a4f7e84) | 融合、深度转换与优化节点源码、许可 | 未运行 |
| Genie Sim PanoRecon | [a3995e925abeffbc5bf68d309a9bdb46af5c1db6](https://github.com/AgibotTech/genie_sim/tree/a3995e925abeffbc5bf68d309a9bdb46af5c1db6/source/geniesim_world) | 模块说明、深度初始化及全景管线源码、模块许可 | 未运行 |
| StreamGS | [2503.06235v2](https://arxiv.org/html/2503.06235v2)；[ICCV 2025 官方论文页](https://openaccess.thecvf.com/content/ICCV2025/html/Li_StreamGS_Online_Generalizable_Gaussian_Splatting_Reconstruction_for_Unposed_Image_Streams_ICCV_2025_paper.html) | 作者论文；本次未核实到可固定的官方实现与权重 | 未运行 |
| S2Tok | [96e8f5885bc3ed9e4f5c5ffb2c66523d92447132](https://github.com/Applied-Intuition-Open-Source/S2Tok/tree/96e8f5885bc3ed9e4f5c5ffb2c66523d92447132)；[论文 v1](https://arxiv.org/html/2610.08978v1) | 论文、推理状态接口、发布说明、代码及权重许可 | 未运行 |
| GlobalSplat | [feb3fd7f7a6a8a9fafcb0ede5c314cd995cdf55b](https://github.com/R-Itk/globalsplat/tree/feb3fd7f7a6a8a9fafcb0ede5c314cd995cdf55b)；[论文 v2](https://arxiv.org/abs/2604.15284v2) | 论文、前向接口、发布说明、许可与额外条款 | 未运行 |

“已检查源码”指下文列出的关键接口，不代表完成上游仓库的全面审计。论文和 README 的性能数据属于作者报告，本仓库不将其登记为本地实验结果。

## 3. 初始化与深度候选

### 3.1 SHARP：单图高斯初始化

**已核实。** 官方定位是从单张图像预测有绝对尺度的高斯表示，用于附近视点合成；默认输出遵循 OpenCV 相机坐标。`RGBGaussianPredictor.forward` 有可选 `depth` 参数，用于预测深度与参考深度的局部尺度对齐，但代码注释建议推理时不提供该参考深度。接口存在不等于发布模型对任意外部预测深度已经过验证。[官方说明](https://github.com/apple-aiml-research/ml-sharp/blob/aed6527499ef91cba3b54c18d49a870f25947190/README.md)、[预测器](https://github.com/apple-aiml-research/ml-sharp/blob/aed6527499ef91cba3b54c18d49a870f25947190/src/sharp/models/predictor.py)。

**可借鉴。** 在许可满足后，将逐帧 SHARP 作为固定的初始高斯生成器，保留原始输出供消融对比。尺度、协方差、四元数、颜色空间与坐标变换须通过明确契约转换。

**不能推断。** 单图的新视角表现不能证明多帧高斯可直接合并；高斯中心、表面覆盖、遮挡后方内容和外观可能跨帧冲突。本阶段不把可选深度参数当作已验证的 DP2 插槽。

### 3.2 Depth Pro 2-L：多帧深度先验

**已核实。** 此处是第二代 Depth Pro 2 的 L 型号。`infer` 接受单图或同一片段的多帧；多帧共同注意并共用一个焦距。公开接口接收 `f_px`，不接收外部位姿或完整内参矩阵；返回 `inverse_depth`、`depth`、`focallength_px`，没有独立置信度输出。`depth` 是米制相机 Z 深度，转换为 `f_px / (width * inverse_depth)`。官方说明的默认视频窗口为 16 帧、重叠 4 帧，并另有因果流模式。[模型接口](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/aa389cbb0b74341664ed663a0966c859905cd6c0/src/depth_pro_2/model.py)、[使用与输出定义](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/aa389cbb0b74341664ed663a0966c859905cd6c0/README.md)。

**可借鉴。** 作为待测量的深度先验，支持逐帧全局尺度校准和重投影一致性筛选。已知相机应使用去畸变后的正确焦距；推理窗口、图像缩放及上下文帧必须写入运行记录。评估帧不得进入重建帧的 DP2 上下文。

**不能推断。** 时间稳定性不是跨视图几何真值，也不能替代位姿、回环或独立深度评估。不能把有效值掩码称为网络置信度，不能把 DP2 自身作为评估 DP2 校准收益的唯一真值。窗口默认值不是本设备的时延或显存保证。

### 3.3 ACCORD：尺度与视图一致性约束

**已核实。** ACCORD 用 OpenVINS 跟踪相机，DA3 处理关键帧窗口；滤波器点为窗口提供米制锚定，子图通过位姿图与回环定位，重叠区域再参与地图校正。它的产物是稠密米制地图，不是 SHARP 高斯融合器。[固定版说明](https://github.com/be2rlab/ACCORD/blob/0a90e98ec793339c8a72ebf7e57540bb74e4b899/README.md)、[作者系统图与方法说明](https://be2rlab.github.io/accord-website/)。

**可借鉴。** 将独立米制姿态/锚点和深度预测分开；用视图重叠检查残差，不以每帧“看起来尺度正确”代替整体一致性。阶段一固定姿态，可减少相机误差和高斯误差的混淆。

**不能推断。** DA3 与 DP2 的接口及条件不同，不能只替换权重文件；ACCORD 地图收益也不能直接转化为高斯渲染收益。若复用其实现，GPL 与各权重条款须分别处理。

## 4. 直接相邻的 SHARP 工程实现

### 4.1 camera-comfyUI：逐帧 SHARP、世界坐标与融合

**已核实。** `VideoToFusedSplats` 选择关键帧并调用 SHARP；提供外部深度时，用投影位置上的参考深度与高斯中心 Z 的中位数比值得到每帧一个尺度，同时缩放中心和高斯尺度；随后把 world-to-camera 位姿取逆，将高斯放入世界系，进行体素合并。`SplatPolish` 提供固定相机下的渲染优化。[GS_nodes.py](https://github.com/Alexankharin/camera-comfyUI/blob/a2f48da1a4a282eae08966c58ca243658a4f7e84/GS_nodes.py)。

**可借鉴。** 这是与阶段一流程最接近的可读实现，可用于核对全局尺度变换、融合预算和消融边界。本仓库当前没有复制或执行其代码。

**限制。** 该深度入口执行全局尺度校准，不执行逐像素深度纠正；中位数也不能自动解决错遮挡或错误对应。其 `pose_nodes.py` 将 VGGT 的 Z 深度转换成径向深度，而上述融合比值使用相机 Z；接入时必须核对并转换语义，不能凭节点名直接连接。[pose_nodes.py](https://github.com/Alexankharin/camera-comfyUI/blob/a2f48da1a4a282eae08966c58ca243658a4f7e84/pose_nodes.py)。没有本仓库复现实验证明其端到端质量或可扩展性。

### 4.2 Genie Sim PanoRecon：外部深度参与 SHARP 初始化

**已核实。** `geniesim_world` 从单张等距柱状全景图产生深度及六个立方体面，再逐面预测并合并高斯。`DepthGuidedRGBGaussianPredictor` 可用外部 `depth_gt` 覆盖两层初始化深度，然后继续经过原有特征、残差预测和高斯组合；因此最终高斯中心并非被硬性约束在输入深度上。DA360 的相对深度与米制深度另有尺度处理。[模块说明](https://github.com/AgibotTech/genie_sim/blob/a3995e925abeffbc5bf68d309a9bdb46af5c1db6/source/geniesim_world/README.md)、[预测器](https://github.com/AgibotTech/genie_sim/blob/a3995e925abeffbc5bf68d309a9bdb46af5c1db6/source/geniesim_world/src/geniesim_world/predictor.py)、[全景管线](https://github.com/AgibotTech/genie_sim/blob/a3995e925abeffbc5bf68d309a9bdb46af5c1db6/source/geniesim_world/src/geniesim_world/cli_pano.py)。

**可借鉴。** 外部深度初始化是有公开代码的研究方向；可在后续独立消融中检查其相对全局尺度校准的收益。

**不能推断。** 同一光心的全景分面，未覆盖移动相机的平移视差、跨时刻遮挡和长轨迹回环。该项目不能作为“DP2-L + SHARP 视频完整重建”已成功的证据，MPL 模块许可也不能覆盖 Apple 模型权重。

## 5. 学习式场景融合路线

### 5.1 StreamGS：对应关系引导的跨帧特征聚合

作者提出从无位姿图像流逐步预测及聚合高斯，以相邻帧对应关系做内容自适应修正，并在特征空间合并冗余。它提供的是“匹配应参与几何与冗余处理”的方法证据，不是现成的 SHARP/DP2 连接器。本次仅核验 [论文 v2](https://arxiv.org/html/2503.06235v2)，未核实到可固定的作者官方实现和权重；不将论文速度或泛化结果登记为本项目可复现能力。

### 5.2 S2Tok：持续更新的空间状态

S2Tok 维护可增大的场景 token 状态，用新帧更新已有表示，并学习哪些新区域值得扩充；每一步从当前状态重新解码累计高斯。公开 `SceneMemory` 与 `step(image, memory)` 接口体现这一设计。固定版 README 提供推理代码及权重链接，但训练、评估代码仍标为待发布，输出相对第一相机且保留全局尺度不确定性。[论文 v1](https://arxiv.org/html/2610.08978v1)、[状态与单步接口](https://github.com/Applied-Intuition-Open-Source/S2Tok/blob/96e8f5885bc3ed9e4f5c5ffb2c66523d92447132/s2tok/model.py)、[发布说明](https://github.com/Applied-Intuition-Open-Source/S2Tok/blob/96e8f5885bc3ed9e4f5c5ffb2c66523d92447132/README.md)。

可借鉴的是“更新已有区域”和“增加新区域”应分开设计。不能据此认为给冻结 SHARP 附加一组全局特征，就等价获得已训练的状态维护能力。其代码和权重均有非商业限制，见许可清单。

### 5.3 GlobalSplat：先融合全局表示，再解码高斯

GlobalSplat 将多视图整合到固定规模的全局场景 token 后再解码高斯，使高斯预算不随输入视图数线性增长。公开前向接口使用图像、内参及 camera-to-world；每次从初始场景 token 构造状态，并非上述 S2Tok 式的持续在线记忆。仓库公开训练/评估入口和预训练权重链接。[论文 v2](https://arxiv.org/abs/2604.15284v2)、[前向接口](https://github.com/R-Itk/globalsplat/blob/feb3fd7f7a6a8a9fafcb0ede5c314cd995cdf55b/globalsplat/model/globalsplat.py)、[发布说明](https://github.com/R-Itk/globalsplat/blob/feb3fd7f7a6a8a9fafcb0ede5c314cd995cdf55b/README.md)。

可借鉴的是场景级预算和在解码前消除冗余的目标。不能把它视为无需训练的通用 SHARP 融合后处理，更不能把其数据集结果直接推断为仓库环境表现。公开代码及模型卡均有非商业条款。

## 6. 第一阶段需要回答，而文献尚未替本项目回答的问题

1. 在同一实测位姿、同一输入帧和同一高斯预算下，DP2-L 的逐帧尺度校准是否比原始 SHARP 初始化稳定改善重叠表面和保留视图？
2. 全局尺度处理后，剩余误差来自局部几何、遮挡、相机标定还是外观？每一类需要独立诊断，不能统称“融合失败”。
3. 去冗余是否减少双层表面和资源消耗，同时保留细杆、边缘及低纹理区域？更少高斯不是单独的成功条件。
4. 结果是否依赖场景、帧顺序或窗口边界？所有比较必须使用预先固定的重建/验证/测试划分和同一资源口径。
5. 若换用许可适配的深度或高斯后端，实验能否仍测量相同问题？替代后端构成新实验条件，不能继承 Apple 模型尚未获得的性能结论。

这些问题目前全部处于待实验状态。阶段一完成的标准由实验计划定义，不以第三方演示视频、源代码可运行或单个漂亮视角代替。

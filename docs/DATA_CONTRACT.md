# 数据与坐标契约

状态：**第一阶段的设计规范。导入器已按本文做 schema 与几何检查，并用合成夹具测试；模型没有运行，真实采集也没有通过这些检查。** 本文中的“必须”仍然约束执行器；合成测试通过不等于真实数据已验收。执行范围与第三方条件见 [THIRD_PARTY.md](THIRD_PARTY.md)，代码范围见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)。

本项目研究：给定标定后的真实静态场景视频与固定的度量相机轨迹，能否利用预训练模型初始化一个可由多视角共同优化的高斯场景。相机轨迹有测量或估计误差，不能因为以米为单位就称为真值。

## 1. 第一阶段的输入边界

| 项目 | 契约 |
|---|---|
| 场景 | 静态真实场景；动态人、机器人、屏幕变化和明显反光异常区域单独标记 |
| 相机 | 单相机、固定镜头；第一阶段不混合主摄、超广角、数字变焦或镜头切换 |
| 图像 | 有实际像素的 RGB 帧；相机标定对应保存后的像素，不只对应预览画面 |
| 标定 | 每帧关联 `K`、畸变与图像变换记录，最终转换到统一针孔相机 |
| 位姿 | 固定 `T_world_camera`，平移单位米；第一阶段不联合优化位姿 |
| 时间 | 单调时钟和明确时间基准；每帧有对应的位姿采样或可追溯插值记录 |
| 深度 | 可选的传感器深度与模型预测分开记录；模型深度不能标成实测真值 |
| 数据拆分 | 场景层的 `dev/test` 与图像层的 `source/validation/test` 分开存储 |
| 起始规模 | 每场最多 48 个 source 关键帧，留出视角目标起点为 24 个；最终精确 ID 在运行前冻结 |

48、24 和 canonical 图像长边 1280 是本研究的**起始实验设置**，不是 SHARP 或 Depth Pro 2 的能力限制。开发场景允许调整；进入正式测试前，所有方法使用同一份冻结配置和精确帧清单。

明显运动模糊、曝光跳变、滚动快门形变和追踪中断必须记录。第一阶段不会假装单个刚体位姿可以解释任意逐行曝光形变，也不以模型生成的图片补充真实观测。

## 2. 标识、不可变输入与最低元数据

原始图片、原始轨迹、原始标定和 split manifest 只读。每次运行记录输入文件 SHA-256、源码提交、模型权重哈希、完整配置、随机种子与输出目录。文件名可以改变，`scene_id`、`frame_id` 和内容哈希不能靠文件遍历顺序隐式推导。

版本0.1的序列化结构以 [scene_manifest.json](../examples/scene_manifest.json) 为格式示例：顶层 `scene_split`，帧内 `split`，`pose_provider` 保存轨迹来源。示例的 `kind=format_example`、合成矩阵、空哈希和占位路径只用于展示格式，**不可作为数据集或可执行实验**。真实运行记录必须补齐实际文件、哈希、变换与验证结果，并使用明确的非示例类型。

### 2.1 场景记录

| 字段 | 类型或单位 | 含义 |
|---|---|---|
| `schema_version` | string | 当前为 `0.1`；不兼容修改必须升版 |
| `kind` | string | 区分 `format_example` 与实际数据记录，示例永不进入执行器 |
| `scene_id` | string | 全数据集唯一 |
| `scene_split` | `dev` 或 `test` | 开发场景与正式测试场景的隔离 |
| `camera_id` | string | 实际物理相机和镜头身份 |
| `world_frame_id` | string | 全场景共用的坐标系身份 |
| `units.distance` | `m` | 不接受隐式厘米、毫米或任意 SfM 尺度 |
| `camera_axes` | `opencv_optical` | x右、y下、z前 |
| `transform` | `T_world_camera` | 相机到世界、列向量约定 |
| `pixel_coordinates` | `integer_center` | 整数是像素中心 |
| `depth_representation` | `z_depth` | 光轴Z-depth；单位另由 `units.distance` 指定 |
| `pose_provider.name` | string | 例如采集时 ARKit、外部跟踪系统或冻结的 SfM 结果；实际记录另附版本/哈希 |
| `pose_provider.metric_scale` | boolean | producer是否声明度量尺度；true不能代替其来源与精度证据 |
| `pose_provider.used_eval_rgb` | `true/false/unknown` | 上游位姿估计是否读过当前留出帧的 RGB |
| `pose_status` | string | `measured_or_estimated_fixed`；只有独立证据才允许标为 reference |
| `clock_domain` | string | 图像与位姿的时钟域及已应用的时间偏移 |
| `split_policy.split_sha256` | string | 冻结拆分内容的哈希；不把该哈希字段自身放进被哈希的内容 |
| `calibration_sha256` | string | 标定及变换记录的哈希 |

### 2.2 帧记录

| 字段 | 类型或单位 | 含义 |
|---|---|---|
| `frame_id` | string | 唯一且跨处理步骤稳定 |
| `split` | `source/validation/test` | 重建输入、开发选参视角、正式评测视角 |
| `timestamp_s` | float64 seconds | 指定单调时钟域内时间，不是格式化墙钟字符串；保存原始精度信息 |
| `raw_timestamp_ns` | optional integer | 来源支持时额外保留精确原始纳秒时间；记录到 `timestamp_s` 时钟域的转换，不能指向不同事件 |
| `timestamp_kind` | enum/string | API capture timestamp、曝光中点等；未知时不可补写为曝光中点 |
| `rgb_path`, `rgb_sha256` | string | 原始保存帧及哈希 |
| `width_px`, `height_px` | positive integer | 实际解码后的像素尺寸 |
| `K` | 3×3 float64 | 对应本记录图像网格的针孔内参 |
| `T_world_camera` | 4×4 float64 | 列向量约定的相机到世界变换 |
| `distortion_model`, `distortion_params` | explicit model + numbers | 原始相机模型；已矫正帧也保留原标定引用 |
| `image_transform` | ordered operations + matrices | 去畸变、旋转、缩放、裁剪及像素中心约定 |
| `pose_timestamp_s` | float64 seconds | 实际匹配到的位姿时间；需要时同时保存原始整数时钟值 |
| `pose_interpolation` | explicit rule or `none` | 若插值，保存两端样本 ID、时间与方法 |
| `tracking_status` | string | 正常、受限、重定位、丢失等来源状态 |
| `exposure_metadata` | optional structured object | 曝光、增益、白平衡、HDR和颜色转换信息；未知须显式表示 |
| `mask_refs` | named references | 图像有效区、动态区、反光区等，不复用一个含义不明的 mask |

矩阵在 JSON 中按“行的数组”序列化，数学运算使用列向量。Swift `simd` 的内存布局不能直接当作 JSON 行序；导出器必须显式重排并用已知投影验证。`preprocess.operation_order`、`preprocess.undistort` 和 `preprocess.canonical_width_px/height_px` 声明预处理，帧记录中的尺寸与K仍必须对应其引用图片的实际网格。

## 3. 唯一的相机与位姿约定

canonical 相机使用右手光学坐标：**x 向图像右侧，y 向图像下侧，z 向镜头前方**。像素原点是左上角像素中心，`u` 向右、`v` 向下。像素中心的整数坐标范围分别为 `0…W−1`、`0…H−1`。

定义：

$$
X_w=T_{wc}X_c,\qquad T_{cw}=T_{wc}^{-1}
$$

$$
T_{wc}=\begin{bmatrix}R_{wc}&t_{wc}\\0&1\end{bmatrix},
\quad R_{wc}^{\mathsf T}R_{wc}=I,\quad \det R_{wc}=+1
$$

本项目字段 `T_world_camera` 始终表示上式的 `T_wc`。传给以 `viewmats` 或 extrinsics 表示世界到相机的渲染器时显式求逆。禁止根据变量名 `pose`、`extrinsics` 猜方向。

投影与反投影：

$$
\tilde p\sim KX_c,\qquad X_c=zK^{-1}[u,v,1]^{\mathsf T}
$$

位姿的世界坐标可以选用 ARKit 世界或其他固定参考系，但一个场景只能有一个 `world_frame_id`。坐标系重置、重定位跳变或不同 session 必须切段或保存显式的段间变换，不能直接拼接轨迹。

### 3.1 ARKit 转换

Apple 文档说明 ARKit 相机坐标随设备方向保持固定，采用 x 向右、y 向上、z 朝屏幕侧的约定；其具体图像方向应结合捕获的像素缓冲区处理。[A1]、[A2]

当原始像素缓冲区与标定的传感器方向已对齐时，定义：

$$
C=\operatorname{diag}(1,-1,-1,1),\qquad
T_{wc}^{\mathrm{optical}}=T_{wc}^{\mathrm{ARKit}}C
$$

这里 `C` 将 optical 相机向量转换到 ARKit 相机坐标。若图片随后转成竖屏或旋转 90°，还要应用下节的相机旋转和内参更新。**不能同时在图片、位姿与渲染器里重复翻转 y/z。**

导入验证至少包含：相机前方的点得到 `z>0`、图像右侧点反投影后位于相机 x 正方向、一个有明确角点的静态物体跨帧投影一致。若用显示用的 `displayTransform`，必须说明它包含的屏幕旋转与裁剪；显示变换不是另一个相机标定。

### 3.2 图片旋转不能只改宽高

将原图顺时针旋转 90°：

$$
u'=H-1-v,\quad v'=u,\quad
Q=\begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix}
$$

对应 `W′=H`、`H′=W`，`fx′=fy`、`fy′=fx`、`cx′=H−1−cy`、`cy′=cx`，并更新：

$$
T'_{wc}=T_{wc}\operatorname{diag}(Q^{\mathsf T},1)
$$

更一般地保存图像齐次变换 `A` 与相机旋转 `Q`，满足 `K′Q=AK`。只给 K 左乘图片旋转矩阵而继续假设其为普通正对角针孔 K，会使一些模型或渲染器误用内参。

## 4. 缩放、裁剪、去畸变与颜色

### 4.1 缩放时的像素中心

若缩放库采用 half-pixel 约定，`u′=sx(u+0.5)−0.5`；缩放后再从 `(x0,y0)` 开始裁剪时：

$$
K'=\begin{bmatrix}
s_x&0&(s_x-1)/2-x_0\\
0&s_y&(s_y-1)/2-y_0\\
0&0&1
\end{bmatrix}K
$$

若实际库采用 `align_corners=True`，横向比例为 `(W′−1)/(W−1)`，不能照搬 `W′/W` 的 half-pixel 公式。保存实际库、插值方式与变换矩阵，采用的公式必须与图像处理一致。

第一阶段 canonical 起始长边为 1280，保持长宽比、使用一致颜色管线。SHARP 内部仍按官方实现处理为 1536 方形；这一步不能反过来改变 canonical K 或误称为原始图像分辨率。[S2]

### 4.2 去畸变

先依据真实相机模型将输入矫正到针孔视图，再进入模型和几何计算。最终保存 `K_rectified`、矫正图像、有效像素 mask，以及从 raw 到 rectified 的映射版本。

鱼眼不能仅套用一个普通 K 而保留原畸变图像。第一阶段不拼超广角与主摄，避免把镜头模型、镜间外参与多模型差异同时混入实验。

DP2 当前公开接口使用一个 clip 的焦距标量，不能据此认为它接受任意逐帧完整 K 或外参。[D1] 本阶段固定镜头、固定 canonical 标定；如原始 K 变化，必须通过可验证的矫正统一到同一针孔 K，或将序列拆开。不能直接平均 `fx/fy` 或忽略变焦。

### 4.3 颜色

输入 RGB 显式标为 8-bit sRGB（或带完整转换记录的其他来源）。YUV 的矩阵、范围、颜色空间以及 HDR 映射有变化时，视为数据管线变化，不能在同一个对照中静默混用。

共享场景内部颜色统一为 `linear_rgb`。SHARP 原生高斯颜色是 linear RGB，其 PLY 导出为了兼容常见查看器会转换成 sRGB；从 PLY 重新导入后不能再假定它仍为 linear RGB。[S3] 所有方法使用同一渲染、曝光处理及评测颜色约定。

## 5. 时间同步与固定轨迹

图像与位姿优先来自同一个采集事件，例如同一 `ARFrame` 的 captured image、camera transform 和 intrinsics。外部跟踪数据需要保存时钟偏移、漂移处理及插值方法；时间戳单位不允许从数值大小猜测。

若插值，平移线性插值、旋转采用旋转空间插值，保存最大间隔和时间残差。采样间隔、时间偏差和允许丢帧规则在开发场景确定后冻结。重定位、明显追踪中断和越界插值不能用平滑结果掩盖。

`timestamp_s` 或 `raw_timestamp_ns` 都不自动等于曝光中点。滚动快门每行曝光、稳定器裁剪、时间延迟未知时须记录为未知；该误差会进入模型效果的解释边界。

**固定的 measured poses 不是无误差的 poses。** 若上游 ARKit 或 SfM 使用了完整视频，其中可能已包含留出帧的信息。本实验必须报告 `pose_provider.used_eval_rgb`。此时结论是“给定固定轨迹的渲染留出”，不能声称完全未见图像条件下的几何重建。进入本项目后禁止再使用留出 RGB 跑匹配、BA、深度估计或高斯优化。

## 6. 深度契约

### 6.1 只接受显式的 optical-axis Z-depth

统一深度定义为相机 z 轴方向的米制深度；序列化写作 `representation=z_depth`、`unit=m`，文中简称 Z-depth。它不是到相机中心的欧氏距离 ray range。设 `r=K^{-1}[u,v,1]^T` 且 `r_z=1`：

$$
\mathrm{range}=z\|r\|,\qquad z=\mathrm{range}/\|r\|
$$

越靠近广角图像边缘，两者差异越不能忽略。世界 Z、逆深度、归一化 disparity、可视化 PNG 色值也不能直接标为米制 `z_depth`。

DP2 官方 `infer` 明确输出沿光轴的米制 depth，同时返回 focal-normalized inverse depth；其 `inverse_depth` 不能无条件作为 `1/depth` 使用。[D1]

### 6.2 深度记录

| 字段 | 约定 |
|---|---|
| `frame_id`, `grid_id` | 指向确定帧和像素网格 |
| `representation`, `unit` | `z_depth` 与 `m`；原始其他定义另存，不覆盖 |
| `source` | `sensor/predicted/reference/none`；`none`必须与无深度文件相符 |
| `path`, `sha256` | 实际深度文件及哈希；没有深度时显式为null |
| `producer` | 模型、传感器、版本及参数 |
| `model_revision`, `weights_sha256` | 模型输出必须记录 |
| `context_frame_ids` | 生成本深度时模型看到的全部帧 ID |
| `depth` | float32 米；无效值为 NaN，并有独立布尔 mask |
| `valid_mask` | 有限、正值、有效图像区域；不是准确性保证 |
| `reliability_weight` | 可选启发式权重，附计算方法；不伪称概率置信度 |
| `metric_alignment` | 若校准过，保存尺度、拟合区域、来源和残差 |
| `input_hashes` | 原始与 canonical 图像、标定、配置哈希 |

DP2 没有公开提供与这里“逐像素可靠性权重”等价的已校准 confidence 输出。[D1] 无效、天空或无限远深度不得填0再当近处几何，也不得为了方便批处理统一压成一堵远平面。

### 6.3 DP2 窗口与同帧重复预测

source 帧先按时间排序，起始配置为窗口 16、重叠 4；只允许帧记录 `split=source`。窗口参数是本研究配置，不代表最优值。最后不足窗口长度时使用实际 source 帧，不拿留出帧补齐，也不复制帧伪造多视图证据。

同一帧在多个窗口产生深度时，对各像素的有效正深度执行固定的 log-median 合并：

$$
D_i(u)=\exp\left(\operatorname{median}_{w\ni i}\log D_{wi}(u)\right)
$$

如果无有效预测，该像素保持无效。合并不拟合任何留出图像，也不按评测分数选择窗口。所有 `context_frame_ids` 必须完整记录；只保存“输出属于 source 帧”不足以证明没有泄漏。

## 7. 高斯参数契约

### 7.1 物理参数与优化器参数分开

共同的物理表示：

| 字段 | 约定 |
|---|---|
| `mean_world_m` | `[N,3]`，世界坐标均值，米 |
| `scale_std_m` | `[N,3]`，严格为正的标准差，米；不是方差或 log scale |
| `rotation_wxyz` | `[N,4]`，归一化四元数，顺序 w,x,y,z；将高斯局部主轴转到世界 |
| `opacity` | `[N]`，物理 alpha，范围 `[0,1]`；不是 logit |
| `color_linear_rgb` | `[N,3]`，linear RGB；不把 SH coefficient 当 RGB |
| `gaussian_id` | 稳定唯一 ID；优化前后保持追溯 |
| `source_observations` | 所有贡献帧、源像素、层编号、合并关系和准入证据 |

协方差定义：

$$
\Sigma=R(q)\operatorname{diag}(s_x^2,s_y^2,s_z^2)R(q)^T
$$

优化器可使用 `log_scale`、`opacity_logit`、`color_logit` 和未归一化的四元数参数，但 checkpoint 必须标为 `raw_parameterization` 并注明激活。导入导出层负责唯一一次转换，禁止对物理尺度重复 `exp`、对 alpha 重复 `sigmoid`。

PLY 的 `scale_0..2` 常采用 log scale，`opacity` 常采用 logit；这不是读取任意文件时的保证。读取器必须根据 producer/schema 解释，检查物理参数再渲染。SHARP 原生导出代码可作为该格式的明确依据。[S3]

### 7.2 SHARP 两层与 provenance

官方实现输出 `2×768×768` 个高斯；两层共同服务于新视角渲染，不能把它们都视为独立实测表面。[S1]、[S4]

原始高斯至少保留：`source_frame_id`、`source_layer_id∈{0,1}`、`source_grid_uv`、`source_grid_shape`、`original_gaussian_index`、源坐标到世界坐标的变换及尺度。解码器会预测位置偏移，所以源 grid pixel 是来源，不保证等于最终高斯投影位置。

层0可用于构造“第一层可见深度”的尺度诊断，仍要经过 alpha、有效区域与几何检查。层1不能在没有跨视角证据时标为物体背面或碰撞表面。多源融合后保留全部来源或可追溯的汇总记录，不能只留下最后一次更新的帧。

### 7.3 尺度与坐标变换

若源相机高斯需要标量尺度校正 `s_i>0`：

$$
\mu_w=R_{wc}(s_i\mu_c)+t_{wc},\qquad
\Sigma_w=s_i^2R_{wc}\Sigma_cR_{wc}^{T}
$$

**不缩放相机的度量平移 `t_wc`。** 在源相机中心缩放几何后再放进世界，不能把整个世界坐标的点乘 `s_i`。颜色与 alpha 不因刚体变换自动改变。

## 8. 可见性、支持与冲突必须区分

给定候选高斯中心 `X` 和 source 相机，计算相机 z 与投影位置。至少区分以下状态：

| 状态 | 定义 | 处理意义 |
|---|---|---|
| `outside` | z≤0、投影越界或无效图像区域 | 没有观测，不计成功也不计失败 |
| `unknown` | 参考深度无效、alpha不足或落在待排除区域 | 无法判断；不是空空间证据 |
| `supported` | 在参考可见表面附近，满足冻结的绝对+相对深度容差 | 可作为几何支持，还需处理表面身份 |
| `occluded` | 候选在参考前景后方，超出容差 | 该视角看不到它，不能当几何错误 |
| `free_space_conflict` | 候选显著位于已观测表面前方 | 若视角和参考可靠，可作为错误证据 |

容差形式及 mask、alpha阈值在开发场景冻结，例如 `tau(z)=tau_abs+tau_rel*z`。这只是拟采用的规则；数值由独立开发验证决定，不宣称已有误差标定。

高斯是有体积的分布，中心深度检查只是第一阶段近似。细边界、大尺度高斯或不同层可能导致中心检查不足，必须单列诊断。`occluded` 不得直接计为拒绝；动态区不得用于静态几何支持。

公共准入与关联不能读外部 DP2 深度来帮助 SHARP-only 对照。它只使用 source RGB/固定 K,T/公共 masks，以及**各实验臂自身初始化**渲染出的参考深度。DP2 特有可靠性仅属于使用它的尺度或 depth loss 分支，详见 [TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md)。

## 9. 精确拆分与泄漏边界

### 9.1 两级拆分

顶层 `scene_split=dev/test` 控制选参是否接触正式测试场景；帧内 `split=source/validation/test` 控制模型与优化能看到哪些图像。两者不得合并为一个含义模糊的 `train/test` 字段。

运行前写入精确列表，禁止只记录“每五帧留一帧”而依赖后续文件排序：

| 图像集合 | 可以做什么 | 不可以做什么 |
|---|---|---|
| `source` | SHARP、DP2、深度合并、准入、关联、共享高斯优化、训练侧诊断 | 不能在运行后按test成绩增删 |
| `validation` | 仅开发场景的渲染选参和预定诊断 | 不进入模型上下文、匹配、初始化或优化 |
| `test` | 冻结方案后读取真实图像计算评测 | 不进入任何重建模型、重建用特征提取、尺度拟合、曝光拟合或优化 |

测试视角的 K/T 是评测渲染所需输入，应由 evaluator 使用；不能把测试 RGB 的特征偷偷作为“全局一致性”条件。LPIPS等预先规定的评测指标可以在evaluator内部计算图像特征，但这些特征与中间结果不得回流到重建、准入或选参。开发集选参记录也必须保存，避免无限试参后仍称为一次测试。

### 9.2 必须记录或阻断的泄漏

- DP2 窗口包含留出帧，虽然只导出 source 深度。
- 先对全视频提取 DINO/匹配/深度，再删除留出输出。
- 用 test RGB 做 BA、相机校准、颜色校准或 per-image exposure fitting。
- 使用 test depth 选取 SHARP 尺度、几何过滤阈值或最优 checkpoint。
- 将相同图片的另一份压缩、裁剪、旋转版本分到另一个集合。
- 根据测试视角渲染效果手工选择高斯或关键帧。

相邻帧可能非常相似，开发时需报告 source 与留出视角的时间和视角距离，并预先冻结采样规则。目标是可解释的条件渲染评估，不通过大量近重复帧夸大泛化。

## 10. 导入完成的判定

数据进入模型前，需要产生可审查的验证报告：帧 ID 与哈希、集合不相交、所有上下文权限、K/T有限性与刚体性、像素变换一致性、相机前向与跨帧角点投影、深度定义及空值、时间同步残差、轨迹中断和颜色管线。任何影响度量坐标的未知项必须显式列出，不能用“模型应该能适应”作为通过理由。

这些是未来实现应提供的验证项；当前仓库没有声称完成数据验证或模型实验。

## 官方依据

以下实现事实固定到所列源码版本；本项目的 schema、切分规则、门槛和窗口合并方法属于研究设计。

- [S1：SHARP predictor，固定提交 `aed6527`](https://github.com/apple-aiml-research/ml-sharp/blob/aed6527499ef91cba3b54c18d49a870f25947190/src/sharp/models/predictor.py)：输入、可选 depth 接口与两层深度路径。
- [S2：SHARP CLI，同一提交](https://github.com/apple-aiml-research/ml-sharp/blob/aed6527499ef91cba3b54c18d49a870f25947190/src/sharp/cli/predict.py)：1536内部处理、焦距使用与反投影。
- [S3：SHARP Gaussian utilities，同一提交](https://github.com/apple-aiml-research/ml-sharp/blob/aed6527499ef91cba3b54c18d49a870f25947190/src/sharp/utils/gaussians.py)：高斯表示、协方差转换、PLY及颜色空间。
- [S4：SHARP 论文 v2，§3](https://arxiv.org/html/2512.10685v2)：模型结构、两层与训练目标。
- [D1：Depth Pro 2 model，固定提交 `aa389cb`](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/aa389cbb0b74341664ed663a0966c859905cd6c0/src/depth_pro_2/model.py)：clip输入、焦距归一化逆深度、optical-axis米制depth与返回字段。
- [A1：Apple ARCamera.transform](https://developer.apple.com/documentation/arkit/arcamera/transform)：ARKit相机方向及世界变换约定。
- [A2：Apple ARCamera.intrinsics](https://developer.apple.com/documentation/arkit/arcamera/intrinsics)：像素内参与左上像素中心原点。

[S1]: https://github.com/apple-aiml-research/ml-sharp/blob/aed6527499ef91cba3b54c18d49a870f25947190/src/sharp/models/predictor.py
[S2]: https://github.com/apple-aiml-research/ml-sharp/blob/aed6527499ef91cba3b54c18d49a870f25947190/src/sharp/cli/predict.py
[S3]: https://github.com/apple-aiml-research/ml-sharp/blob/aed6527499ef91cba3b54c18d49a870f25947190/src/sharp/utils/gaussians.py
[S4]: https://arxiv.org/html/2512.10685v2
[D1]: https://github.com/apple-aiml-research/ml-depth-pro-2/blob/aa389cbb0b74341664ed663a0966c859905cd6c0/src/depth_pro_2/model.py
[A1]: https://developer.apple.com/documentation/arkit/arcamera/transform
[A2]: https://developer.apple.com/documentation/arkit/arcamera/intrinsics

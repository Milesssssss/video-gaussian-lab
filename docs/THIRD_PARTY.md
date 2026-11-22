# 第三方来源、许可与执行前置条件

核验日期：2026-11-22。本文记录固定版本的公开条款及本项目决策，不替任何权利人授予使用许可。

## 1. 当前状态与首要约束

**当前为 `design_only`：本轮只编写方案文档，未下载模型权重、未启动模型、未训练模型、未产生模型实验结果。SHARP 与 Depth Pro 2-L 不是已获商业使用许可的默认后端。**

Apple 的两份 `LICENSE_MODEL` 都把许可限定为非商业科学研究与学术开发，并在范围条款中明确排除商业利用、产品研发（`product development`）及商业产品或服务用途。公司内部 PoC、未收费、private 仓库、仓库名称含 research，均不是条款给出的产品研发豁免。针对本项目面向产品技术路线的用途，执行 Apple 模型前应取得覆盖该用途的书面许可；若主张属于原条款允许的独立非商业科研，也须先记录实际目的和适用性依据，不能只改项目标签。[SHARP 模型条款](https://github.com/apple-aiml-research/ml-sharp/blob/aed6527499ef91cba3b54c18d49a870f25947190/LICENSE_MODEL)、[DP2 模型条款](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/aa389cbb0b74341664ed663a0966c859905cd6c0/LICENSE_MODEL)。

本仓库的文献讨论、独立接口设计和实验预注册不表示已获模型运行许可。引用来源也不授权复制其代码、权重、数据、演示资产或模型输出。当前不提交这些第三方资产。

## 2. 可以继续推进的路线

| 路线 | 可开展的工作 | 进入模型实验的条件 |
|---|---|---|
| A：先完成与模型无关的研究基础 | 固定数据契约和保留视图；设计坐标、深度、协方差转换；用自建解析场景或已获使用权的传感器观测验证几何与指标 | 不依赖受限预训练权重；使用到的库、传感器数据和评估资产仍按实际许可记录 |
| B：获得适用授权后运行 Apple 候选 | 保留本方案的 SHARP + DP2-L 消融设计 | 书面授权覆盖执行主体、内部评估/产品研发等实际用途、精确模型与版本、必要的输出及分发范围；或有充分依据确认活动属于原许可允许范围 |
| C：按接口替换后端 | 深度提供器与高斯初始化器独立可换；可以使用获准深度后端，或从已授权传感器深度自行初始化高斯 | 分别核验精确代码、checkpoint、上游权重和附加条款；冻结新后端及新的实验版本后再运行 |

本文件没有把某个替代网络认证为“可商用”。开源代码标签、权重可下载、某个尺寸的同名模型采用宽松许可，都不能替代对实际使用版本的核验。改变后端后仍可研究同一融合问题，但不能沿用另一模型的精度、速度或显存结论。

## 3. Apple 代码与模型必须分开判断

| 项目与固定源码版本 | 代码条款 | 模型条款与本项目判断 |
|---|---|---|
| SHARP：`aed6527499ef91cba3b54c18d49a870f25947190` | Apple 自定义 [LICENSE](https://github.com/apple-aiml-research/ml-sharp/blob/aed6527499ef91cba3b54c18d49a870f25947190/LICENSE)，不是 MIT/Apache；第三方组件另见 [ACKNOWLEDGEMENTS](https://github.com/apple-aiml-research/ml-sharp/blob/aed6527499ef91cba3b54c18d49a870f25947190/ACKNOWLEDGEMENTS) | [LICENSE_MODEL](https://github.com/apple-aiml-research/ml-sharp/blob/aed6527499ef91cba3b54c18d49a870f25947190/LICENSE_MODEL) 限研究用途，产品研发不在公开许可范围；当前没有覆盖本项目用途的授权记录 |
| Depth Pro 2-L：`aa389cbb0b74341664ed663a0966c859905cd6c0` | Apple 自定义 [LICENSE](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/aa389cbb0b74341664ed663a0966c859905cd6c0/LICENSE)；上游组件见 [ACKNOWLEDGEMENTS](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/aa389cbb0b74341664ed663a0966c859905cd6c0/ACKNOWLEDGEMENTS) | [LICENSE_MODEL](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/aa389cbb0b74341664ed663a0966c859905cd6c0/LICENSE_MODEL) 同样明确限制；不能把上一代 Depth Pro 的说明套用到第二代 L |

两者源代码许可允许若干使用、修改和再分发行为，并保留通知等条件；这不授予绕过模型条款的权利。模型条款对模型的定义不只包含权重，也涉及参数、配置、检查点及相关材料；模型派生和再分发另有范围、说明及署名要求。当前方案不会通过换封装、复制 checkpoint、微调或改仓库可见性来改变这些限制。

官方说明中的候选权重标识如下，仅用于未来核对，没有在本项目下载：

| 候选资产 | 官方发布标识 | 当前锁定程度 |
|---|---|---|
| SHARP | `sharp_2572gikvuh.pt`，见固定版 [README](https://github.com/apple-aiml-research/ml-sharp/blob/aed6527499ef91cba3b54c18d49a870f25947190/README.md) | 已核对名称；尚无本地 SHA-256 或授权记录 |
| DP2-L | `depthpro2-l-tegstu7fim.pt`，见固定版 [README](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/aa389cbb0b74341664ed663a0966c859905cd6c0/README.md)；[官方模型集合](https://huggingface.co/apple/DepthPro2) | 已核对型号及名称；尚无本地 SHA-256 或授权记录 |
| DP2 焦距估计器（仅在需要时） | `depthpro2-fov-ew55zg4kq7.pt`，同一官方说明 | 独立资产，也须记录；已知焦距路线不能无故隐式下载它 |

权重文件名和源码 commit 不能充当权重内容校验值。使用不同格式或分发渠道的权重，也须独立记录 revision、文件名、哈希与对应条款。官方快捷入口可能自动下载权重，因此许可与用途记录应在启动这些入口之前完成。

## 4. 其他相关工作的条款边界

以下是阅读版本的核验，不是已安装依赖清单。源代码是否允许某种使用，与整个模型流水线是否允许同一使用，是两个问题。

| 工作 | 已核实的条款与来源 | 本仓库当前处理 |
|---|---|---|
| ACCORD | 固定版 [LICENSE](https://github.com/be2rlab/ACCORD/blob/0a90e98ec793339c8a72ebf7e57540bb74e4b899/LICENSE) 为 GPL-3.0；[README](https://github.com/be2rlab/ACCORD/blob/0a90e98ec793339c8a72ebf7e57540bb74e4b899/README.md) 明示 DA3 权重另有条款，默认 Giant 为 CC BY-NC 4.0 | 仅参考尺度与一致性思想。GPL 本身不等于禁止商业使用；复制或组合代码、默认权重和其他依赖要分别评估 |
| camera-comfyUI | 固定版 [LICENSE](https://github.com/Alexankharin/camera-comfyUI/blob/a2f48da1a4a282eae08966c58ca243658a4f7e84/LICENSE) 为 MIT，并明确排除 `submodules/ml-sharpt`，后者仍受 Apple 代码与模型条款约束 | 只读关键实现；MIT wrapper 不会解除 SHARP 权重限制 |
| Genie Sim World / PanoRecon | 本模块 [LICENSE](https://github.com/AgibotTech/genie_sim/blob/a3995e925abeffbc5bf68d309a9bdb46af5c1db6/source/geniesim_world/LICENSE) 为 MPL-2.0；[README](https://github.com/AgibotTech/genie_sim/blob/a3995e925abeffbc5bf68d309a9bdb46af5c1db6/source/geniesim_world/README.md) 说明只覆盖其原创源码，SHARP、DA360 等单独授权 | 只参考全景管线和深度注入；未将其或 DA360 判定为整套可商用依赖 |
| StreamGS | 已核对 [论文 v2](https://arxiv.org/html/2503.06235v2)，本次未核实可固定的官方代码/权重及其许可 | 仅方法参考。论文的内容许可不等于实现或权重的使用许可 |
| S2Tok | [LICENSE](https://github.com/Applied-Intuition-Open-Source/S2Tok/blob/96e8f5885bc3ed9e4f5c5ffb2c66523d92447132/LICENSE) 与 [WEIGHTS_LICENSE.md](https://github.com/Applied-Intuition-Open-Source/S2Tok/blob/96e8f5885bc3ed9e4f5c5ffb2c66523d92447132/WEIGHTS_LICENSE.md) 将原创代码和发布权重设为 CC BY-NC 4.0，并列出 ZipSplat、DA3 与训练数据的上游约束 | 仅方法与接口参考；不能作为已获许可的商业替代后端 |
| GlobalSplat | 自有代码采用 [PolyForm Noncommercial 1.0.0](https://github.com/R-Itk/globalsplat/blob/feb3fd7f7a6a8a9fafcb0ede5c314cd995cdf55b/LICENSE) 加 [ADDITIONAL_TERMS.md](https://github.com/R-Itk/globalsplat/blob/feb3fd7f7a6a8a9fafcb0ede5c314cd995cdf55b/ADDITIONAL_TERMS.md)；商业用途需另行书面许可，额外条款涵盖为第三方产品/服务研发的用途；[官方模型卡](https://huggingface.co/Roni-It/globalsplat) 对权重也标明非商业及额外条款 | 仅方法参考；未下载权重，模型卡 revision 尚未锁定。第三方 vendored 代码保持各自许可，不能由此推导整套授权 |

GlobalSplat 的额外条款另有限定用途；后续若采用它，应核对全文，而非只记录一个许可证简称。S2Tok 的权重说明还明确涉及训练所用模型和数据条款，不能因为部分底层代码为 Apache-2.0 就把整体权重标为 Apache-2.0。

## 5. 从方案进入实验的必要记录

首次运行前，应为**实际选中的**后端完成以下记录；未选中的文献工作无需安装或批量下载。

1. **用途与主体：** 谁运行、实验服务什么目的，是否包含公司产品研发、客户演示、输出交付或部署；与许可或书面授权的具体依据对应。
2. **精确资产：** 源码完整 commit、依赖版本、checkpoint 分发地址及 revision、文件名、内容哈希、许可证原文版本。当前未取得的哈希标记为未取得，不编造。
3. **适用范围：** 分别记录代码、权重、上游模型、数据、渲染器和评估模型；一项通过不能自动放行整条流水线。
4. **执行与分发：** 明确可运行环境、可访问人员，以及是否允许发布权重、派生模型、结果图、PLY、数据和报告；只获取实验实际需要的权限。
5. **变更控制：** 修改模型尺寸、来源、格式、运行目的或将结果转入产品时重新核对。运行日志应能追溯该判断和授权范围。

模型输出应保留生成来源与用途记录。本文件不把所有输出一律认定为模型派生物，也不把所有输出默认视为可自由商业使用；具体输出的使用和分发应按实际授权、输入数据权利及适用条款判断。阶段一方案中的成功结果也不会自动解除这些条件。

## 6. 本轮完成与待完成

**已完成：** 固定关键源码阅读版本，核对 Apple 代码/模型许可原文，区分相关项目的自有代码与所依赖的模型条款，记录不依赖受限权重即可继续的研究基础工作。

**待完成：** 确定实际执行路线，记录覆盖实际用途的依据，锁定获准的精确权重及依赖，再开始模型实验。本文件当前不声明商业可用性、不声明模型复现成功，也不以未来可能获得的授权替代当前记录。

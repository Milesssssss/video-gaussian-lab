# Phase 1 执行基础状态

状态日期：2026-11-23。本文记录代码已经实现并经过合成测试的部分。它不是实验报告，不包含 PSNR、SSIM、LPIPS、几何误差、耗时或显存的实测值。

协议登记仍是 [configs/phase1_protocol.json](../configs/phase1_protocol.json)：`execution_enabled=false`，`implementation_exists=false`，`status=draft_not_frozen`。`implementation_exists` 继续表示“已有可执行的正式实验栈”。当前只有契约、几何和阻塞接口，因此该字段保持 false。

## 1. 状态分层

| 层 | 状态 | 证据 |
| --- | --- | --- |
| 设计 | v0.1 方案保留 | `docs/` 与协议 JSON |
| 已实现 | `src/vglab` 中的校验、几何、损失定义、实验臂隔离和正式运行拒绝 | 模块与 `tests/` |
| 已测试 | 合成夹具上的单元测试。示例清单 `kind=format_example`，不是采集 | `python -m pytest` |
| 已测得 | 无 | `measurements` 保持 null；报告模板仍是未运行 |

P1b 的 4 个测试场景 × 5 个优化组 × 3 个种子没有启动。

## 2. 模块与验收差距

| 模块 | 已实现并在合成数据上测试 | 尚未发生 |
| --- | --- | --- |
| `ingest` | schema、划分哈希、source/validation/test 互斥、留出帧不能进入上下文 | 真实视频、文件存在性意义上的采集审计 |
| `calibration` | 投影/反投影、刚体、ARKit `C`、顺时针 90°、half-pixel 缩放、Z-depth 与 range | 真实角点重投影图 |
| `depth` | source 窗口 16/4、不足窗口不补齐、log-median、`fx!=fy` 时拒绝平均、teacher 固定掩码与启发式 q、多 source 支持/自由空间冲突 | DP2 推理与权重哈希 |
| `initialization` | 每帧一个 `s_i`、失败不截断、普通像素足迹高斯、SHARP 放置、注入式 SHARP/DP2 适配器与前置条件门 | SHARP/DP2 前向、真实 torch 桥 |
| `scene` | 可见性四态、准入配额、邻域桶去重；只有层、朝向、颜色和尺度范围都一致才抑制代表，预算不足不复制 | 真实渲染深度上的准入 |
| `render` | 开发用参考渲染器：EWA 2D 投影、front-to-back 合成、期望深度/alpha、解析梯度与相对步 | 与官方光栅化器的单帧对齐、真实分辨率性能 |
| `optimize` | 期望深度、空深度监督、带掩码的 L1/SSIM、相机不可训练、S3/D1 参考深度精化与覆盖率停止规则 | 已验证的高斯光栅化和正式共享优化步 |
| `evaluate` | PSNR 定义、覆盖率、拒绝把预测当几何真值、候选决策规则的纯函数 | 留出图像指标和独立参考 |
| `reporting` / CLI | 阻塞说明；`run-official` 拒绝执行；`demo-depth-constraint` 合成参考演示 | 冻结记录和实验报告 |

S0/S1 读取 DP2 深度、尺度字典或把准入几何来源标成 `dp2` 时会失败。S2/S3 的 DP2 只通过已接受的每帧标量进入。尺度同时乘相机坐标均值和标准差，协方差按 `s_i^2` 变化，相机平移不乘 `s_i`。

G0 为可用时，H2/H4 的候选判定要求 4 个场景都有 F-score。缺任一场景时结论是 inconclusive，不用已有子集的平均代替。G0 为 NA 时几何保持未评测，H4 只按渲染和成本的候选规则返回。去重的朝向点积下限 `0.9659258262890683` 和颜色 L2 上限 `0.15` 是未冻结代码参数。

普通高斯和尺度拟合里的数值在 `initialization/candidates.py`，`frozen=false`。它们不是协议冻结值，也没有在开发场景上选过。

## 3. 可以执行的命令

下列命令有对应测试。它们不启动 SHARP、DP2 或 P1a/P1b。

```text
python -m vglab.cli validate-manifest examples/scene_manifest.json
python -m vglab.cli show-blockers
python -m vglab.cli run-official
python -m vglab.cli demo-depth-constraint --steps 24
```

`validate-manifest` 对格式示例返回 schema 通过且不可执行。`--require-captured` 对同一文件返回非零。`run-official` 返回非零。`demo-depth-constraint` 运行合成参考演示，输出显式标记 `kind=synthetic_reference_demo`、`not_measured=true`，不写协议或结果文件。

## 4. 深度损失的实现选择

空的 teacher 区域或空的实际监督返回 `skipped_empty_depth_mask`，损失为 null。teacher 为空时覆盖率是 null，不是 100% 或 0。有正权重时使用 `mean |log Z_render - log Z_dp2|`，分母是权重和。

[TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md) 的分母写有 `epsilon`，[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) 把 `epsilon` 写在对数内部。两处都是未冻结的数值稳定写法。当前实现与尺度公式一样，只对有限正深度取对数，并且不用 `epsilon` 把空监督变成零损失。正式权重仍待开发集冻结。

## 5. 仍然阻塞模型实验的条件

依据 [THIRD_PARTY.md](THIRD_PARTY.md) 与协议 `entry_conditions`：

- SHARP 固定源码 `aed6527499ef91cba3b54c18d49a870f25947190`，权重文件名 `sharp_2572gikvuh.pt`，本地 `weight_sha256` 为 null。
- Depth Pro 2-L 固定源码 `aa389cbb0b74341664ed663a0966c859905cd6c0`，权重文件名 `depthpro2-l-tegstu7fim.pt`，本地 `weight_sha256` 为 null。
- 两份 `LICENSE_MODEL` 都把公开许可限制在非商业科研，并排除产品研发。本仓库没有覆盖实际用途的书面授权记录。
- 没有 2 个开发场景和 4 个测试场景的真实标定视频、米制轨迹或独立几何参考。
- 本执行环境没有 `nvidia-smi`。没有 GPU 推理被记成已运行。

因此 P1-01 的真实场景登记、P1-03 的模型哈希、P1-05 至 P1-08 的运行和结论都还没有完成。阻塞是前置条件，不是算法失败，也不是阴性实验结果。

## 6. 深度约束参考链路（开发用）

2026-11-23 新增，用于把“DP2 深度 → teacher → 渲染深度 → 残差 → 高斯”这条链在合成夹具上接通。它不是正式渲染器或正式优化器：

| 组件 | 位置 | 已验证内容 | 未验证 |
| --- | --- | --- | --- |
| teacher 构造 | `depth/teacher.py` | 固定 teacher 掩码、启发式 q 的边界、多 source supported/occluded/unknown/free-space-conflict、留出帧拒绝 | 真实 DP2 输出、开发集阈值冻结 |
| 参考渲染 | `render/reference.py` | 单高斯解析值、两层期望深度公式、z 排序、解析深度/opacity 梯度与有限差分一致、相对步与 Jacobi 预条件 | 官方光栅化器等价性、真实分辨率性能 |
| 深度精化 | `optimize/depth_refine.py` | S3/D1 臂限制、固定 teacher 掩码与 K/T 不被改写、空监督与低覆盖停止、合成平面上误差单调下降 | 正式训练器、RGB 与深度的联合优化、实测收敛 |
| 适配器执行层 | `initialization/adapters.py` | 后端注入契约、权重/许可/加速器前置门、层与 grid 校验、单窗口焦距一致、缓存键 | Apple 权重前向、真实 torch 桥 |

`demo-depth-constraint` 的数值是合成夹具上的接线自检：初始深度 2.4 m、teacher 2.0 m 的平面在 24 步内把平均 |log 误差| 从 0.182 降到 1e-5 量级。它不是重建质量、速度或显存结果，也不能外推到真实数据。

`render` 的参考渲染器把高斯按中心 z 全局排序后合成，复杂度 O(N×补丁像素)，只适用于小规模合成夹具；正式实验仍必须换成经过单帧对齐验证的共享渲染器，`execution_enabled` 与 `implementation_exists` 保持不变。

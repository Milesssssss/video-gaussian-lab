# UI 集成说明

## 目标与当前边界

本次将用户上传的前端原型整理到 `ui/`，保留暗色与 cyan 的工作台布局，并把展示分成合成演示、协议状态和本地视频预览。Python 研究代码与 `configs/phase1_protocol.json` 的执行状态不因 UI 集成而改变。

**这是研究工作台的前端入口，尚未接通真实模型推理和共享高斯场景优化。** UI 的演示数据不能填入实验报告，预览成功也不能表示真实重建流程已通过验证。实现、测试、测量仍按[实现状态](IMPLEMENTATION_STATUS.md)区分。

## 原型来源与可追溯性

| 项目 | 记录 |
| --- | --- |
| 来源 | 本次会话中用户上传的 ZIP |
| 原文件名 | `gaussianflow-sharp.zip` |
| 原型标识 | `SHARP · 3DGS Lab`，取自原始 `metadata.json` |
| ZIP 大小 | 4,146,448 bytes |
| ZIP SHA-256 | `842cbb82d815b198488ecacd7008f166a58426424fa32b16fce85f7cf40dec05` |
| 哈希取得方式 | 对上传 ZIP 的原始字节实际计算 SHA-256 |
| 集成基线 | 远程 `main` commit `3818dead5141452b6832f08b37ae6e5abed14515`，tree `bcef42717ec6c70a6fe44dc15481228f27cf622f` |
| 本次目标目录 | `ui/` |

原 ZIP 与解包参考目录不提交到 Git。上面的哈希标识原始上传物，**不代表整理后 `ui/` 文件树的哈希**。后续变更由 Git commit 和 npm lockfile 追踪。

来源记录不替代第三方授权，也不重新授予上游模型或依赖的使用权。尤其不能因本仓库是研究文档或 private 仓库，就将 Apple 模型视作默认可用于公司产品研发；执行前仍按[第三方许可记录](THIRD_PARTY.md)落实用途与授权。

## 导入与裁剪决定

| 原型部分 | 本次处理 | 理由 |
| --- | --- | --- |
| 工作台布局、导航、侧栏、时间轴、三维视口 | 整理为 `ui/` 内的组件与样式 | 保留交互结构，统一数据来源和状态表达 |
| 合成场景、演示相机和关联计算 | 放在明确的合成演示路径，界面持续标注来源 | 适合验证交互；不能解释为真实视频的 SHARP、DP2 或全局特征结果 |
| 模型对比与性能展示 | 使用真实协议状态；没有实测值时显示未运行 / 未接入 | 手写误差、PSNR 或控制参数的单调变化不能证明算法增益 |
| `server.ts` 与 `apple_worker/` | 不导入 | 本次不建立上传与推理服务；原 worker 的 Depth Pro 安装路径与 DP2-L 标签也不等价 |
| 浏览器神经深度及原视频重建链路 | 不导入 | 原实现调用 Depth Anything，并用人为区间转换相对预测，不能标为 Apple DP2-L 或有依据的米制深度 |
| 原 PLY 转高斯路径 | 不导入 | 原导入没有完整保留高斯旋转等物理参数，不能作为正式 3DGS 数据接口 |
| 演示点云导出 | 仅导出合成点的位置与颜色，并在文件名和内容中标识来源 | 这是可查看的演示点云，不承诺完整高斯场景的往返保真 |
| 原 AI Studio 环境说明、生成截图与旧依赖锁文件 | 不作为新工程入口 | 新入口使用 npm lockfile，不要求 API key，不携带无关生成环境配置 |

原型中的参考相机可以作为**合成世界的已知真值**参与演示，但这种已知量不能套用到用户选择的视频，更不能作为真实实验的独立几何参考。正式研究的 S0–D1 实验臂定义仍以[实验协议](EXPERIMENT_PROTOCOL.md)为准，演示控制模式不能冒名对应已执行的实验臂。

## 三种数据的隔离

| 数据类别 | 本次 UI 的行为 | 不能据此推断 |
| --- | --- | --- |
| 合成预设 | 在浏览器计算并展示场景、参考相机与示意关联 | 模型已下载、推理已运行、真实场景质量或性能已测得 |
| 本地视频 | 读取元数据与预览帧；不上传，不请求模型 | 已获得 K、米制位姿、深度、帧级同步、重建场景或独立 GT |
| 仓库协议快照 | 读取构建所用版本的 `configs/phase1_protocol.json`，展示状态与实验定义 | 存在在线 runner、状态正在实时同步，或用户已具备模型执行许可 |

本地预览与演示场景分别存储。预览文件名、时长或缩略图不会使合成场景变成该文件的重建结果。元数据不可用时保留错误或缺失状态，不能补造相机参数或深度。

当前没有 UI 到 Python CLI 的启动通道。`execution_enabled=false` 继续由仓库协议保留；前端不修改协议，不提供绕过准入条件的开关，不以完成动画模拟模型任务状态。

## 运行与 CI

使用 Node.js 24；版本记录在 `ui/.nvmrc`。从仓库根目录执行：

```bash
cd ui
npm ci
npm run dev
```

开发地址为 [http://localhost:5173](http://localhost:5173)。`npm run check` 做类型检查，`npm test` 执行前端单元测试，`npm run build` 做类型检查与生产构建。完整说明见 [UI README](../ui/README.md)。

独立 workflow [`.github/workflows/ui.yml`](../.github/workflows/ui.yml) 只授予 `contents: read`，在 `ui/**`、`configs/phase1_protocol.json` 或该 workflow 自身变更时触发。它使用一个 Ubuntu / Node.js 24 job，顺序执行 `npm ci`、`npm run check`、`npm test`、`npm run build`，然后运行 `npx playwright install --with-deps chromium` 与 `npm run test:browser`。既有 Python workflow 保持不变。

workflow 最后的截图保存步骤使用 `always()`，将 `ui/test-results/**/*.png` 保存为 `ui-browser-screenshots` artifact；`if-no-files-found: ignore` 允许浏览器未启动或尚未产生截图时跳过保存。保存步骤不改变前面测试的失败状态，截图存在也不等于所有断言通过。

GitHub Actions 引用使用本次已核对存在的官方版本：[checkout v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1)、[setup-node v7.1.0](https://github.com/actions/setup-node/releases/tag/v7.1.0) 与 [upload-artifact v7.0.2](https://github.com/actions/upload-artifact/releases/tag/v7.0.2)。这些版本号不是本项目 workflow 已执行成功的证明。

### 验证记录

2026-11-22 核验代码提交：`107e91ab403fbf3c3f13e6359ca19b677593a0af`。

- [UI CI run 45009815054](https://github.com/Milesssssss/video-gaussian-lab/actions/runs/45009815054)：依赖安装、类型检查、单元测试、生产构建、浏览器测试和截图上传均成功。
- [Python CI run 45009815035](https://github.com/Milesssssss/video-gaussian-lab/actions/runs/45009815035)：既有回归测试通过；本 PR 没有修改 Python 研究逻辑。

| 检查 | 结果 | 验证范围 |
| --- | --- | --- |
| `npm ci`、`npm run check` | 通过 | lockfile 可安装，TypeScript 检查通过 |
| `npm test` | 16 / 16 通过 | 合成计算、协议展示、本地视频参数和点云导出来源 |
| `npm run build` | 通过 | 生产构建完成；Three.js chunk 仍有体积提示 |
| 桌面 / 手机尺寸浏览器检查 | 6 / 6 通过 | 场景控制、点云下载、六个实验臂、键盘探针、本地视频解码和退出行为 |
| 三维视口可见像素断言 | 两种尺寸均通过 | 实际截图中央采样区中，最大 RGB 分量大于 80 的像素占比为桌面 0.4986、手机 0.7737，均大于 0.01 的防空白阈值 |
| 人工截图检查 | 完成 | 工作台、实验对照、深度检查和视频预览已复核；首帧修复后重新确认桌面和手机工作台画面 |
| 真实 SHARP / DP2 / 共享优化 / 独立指标 | 本次未运行 | 不能据此报告研究结果 |

首轮交互测试通过后，人工截图发现工作台在首次绘制完成前被截成空白。修复增加 `pending / ready / lost / error` 绘制状态，在 GPU 完成且画面可呈现后才发布 `ready`；相同尺寸不再重复调用 `setSize()` 清空画布。测试同时检查实际画面像素，避免仅凭 canvas 已挂载就判定渲染成功。上述像素占比只用于识别空白视口，**不是重建质量指标**。

浏览器用例位于 [`workbench.spec.ts`](../ui/tests/workbench.spec.ts)，使用生产构建、Playwright 1.62.1 和 Chromium，在 1440 × 1000 与 390 × 844 两种尺寸运行。测试视频是 1.2 秒的合成 WebM fixture，不是用户采集。手机配置是尺寸 / 触控模拟，不代替真实设备验证。本地运行环境不能启动 Chromium，因此浏览器证据来自上述 GitHub CI。

[桌面工作台截图](ui/workbench-desktop.png)与[手机工作台截图](ui/workbench-mobile.png)保留了 CI 生成的原始 PNG。完整 artifact `ui-browser-screenshots` 的 SHA-256 为 `58f8db5a063bcfa6b95c6ad1b99757cae22ed1261b61f5d8660d39a3f245ba2c`，包含 8 张页面截图和 2 张用于防空白检查的视口截图。

单元测试使用 `node --import tsx --test`，避免 `tsx` CLI 额外建立 IPC。Three.js 视口为懒加载 chunk，此次 CI 构建为 592.99 kB、gzip 148.78 kB；保留默认超过 500 kB 的体积提示，没有调高阈值掩盖提示。

## 下一步：正式数据接口

以下为后续接入要求，**本次没有实现这些后端接口**。接口应复用[数据契约](DATA_CONTRACT.md)、[实验协议](EXPERIMENT_PROTOCOL.md)与实际 runner 状态，避免在前端创造另一套实验定义。

1. **先接入经过校验的场景清单。** 保留明确的场景与帧 ID、来源文件哈希、图像尺寸、畸变处理、同步信息、`K` 和固定米制 `T_world_camera`。先冻结 `source / validation / test`；预览时生成的时间戳不能自动成为已同步的正式帧记录。
2. **每种数据标明来源。** 传感器观测、模型预测、独立参考和合成演示必须可区分。深度声明光轴 `z_depth` 与米单位、有效掩码、预测模型版本和上下文帧；没有独立几何参考时保持缺失，不能用预测深度填 GT。
3. **读取真实运行与报告。** 先建设只读的 manifest、运行状态及报告适配层，携带配置哈希、代码 / 模型版本、失败状态、耗时与指标来源。只有实际测量才能标成 measured；尚未实现的 runner 不返回虚构进度和结果。
4. **完整保留高斯物理量。** 正式场景接口遵守数据契约的中心位置、`scale_std_m`、`rotation_wxyz`、`color_linear_rgb`、透明度与 `source_observations`。导入器必须明确 log / logit 激活、颜色空间和坐标变换；完成往返与参考渲染核验后，才加入正式 3DGS 导入导出。
5. **在后端满足准入后才开放任务操作。** 许可、环境、数据划分与实际 renderer / optimizer 的验证仍是前置条件。UI 不通过按钮、协议副本或默认相机代替这些条件。任务启动、取消与结果管理属于后续独立集成。

Three.js 按实际使用的成员显式导入，不使用全局 `THREE.*`。前端自己的演示渲染和导出格式也不能自动视为研究渲染器的等价实现。

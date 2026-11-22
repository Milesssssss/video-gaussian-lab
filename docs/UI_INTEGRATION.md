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

下表记录 2026-11-22 本次集成的本地验证结果。类型检查、单元测试和构建已实际完成；浏览器在启动阶段受环境限制，分支 CI 仍待核验。不能用历史 Python CI 的成功替代本次前端结果。

| 检查 | 当前记录 | 结果能支持的结论 |
| --- | --- | --- |
| `npm ci` | 分支 CI 待核验 | 仅在实际安装成功后确认 lockfile 可安装 |
| `npm run check` | 本地通过 | 当前 TypeScript 类型检查通过 |
| `npm test` | 本地 16 / 16 通过 | 合成演示与前端辅助逻辑的单元测试通过 |
| `npm run build` | 本地通过；有 chunk 体积提示 | 当前生产构建可完成，不代表浏览器交互已通过 |
| 桌面 / 手机尺寸浏览器检查 | 6 个用例均在本地 Chromium 启动阶段失败，未进入页面断言；分支 CI 待核验 | 尚不能判断这些浏览器用例是否通过 |
| 本地视频元数据、抽帧、取消与失败路径 | 已编写浏览器用例；本地未进入断言，分支 CI 待核验 | 尚未验证浏览器中的解码与交互结果 |
| GitHub UI workflow | 尚未取得本分支 / PR 的运行证据 | 仅在实际 job 成功后计为 CI 通过 |
| 真实 SHARP / DP2 / 共享优化 / 独立指标 | 本次范围外，未运行 | 不能报告研究结果 |

单元测试通过 `node --import tsx --test` 执行，避免 `tsx` CLI 本身额外建立 IPC。生产构建中的 Three.js 视口为懒加载 chunk，本次输出 591.99 kB、gzip 148.45 kB，仍触发默认超过 500 kB 的提示；这不是构建失败，后续是否继续拆包需结合实际加载表现决定。

浏览器测试定义在 [`workbench.spec.ts`](../ui/tests/workbench.spec.ts) 和 [`playwright.config.ts`](../ui/playwright.config.ts)，使用 Playwright 1.62.1 与 Chromium。3 个场景分别在 1440 × 1000 和 390 × 844 两种尺寸运行，共 6 个用例；手机配置是浏览器尺寸 / 触控模拟，不能代替真实设备验证。用例检查合成场景控制与点云来源、六个实验臂及无实测结果状态、键盘探针，以及 1.2 秒合成 WebM 的本地预览、无外部上传和退出行为。测试视频是专用合成 fixture，不是用户采集。

本地尝试使用 `/usr/bin/chromium`，进程在页面建立前受到 Unix socket 权限限制，因此 6 个失败不能解释为 UI 断言失败，也不能视为浏览器通过。分支 CI 安装 Playwright 所需 Chromium 后执行同一套用例，结果与截图仍需实际检查；截图用于人工审图，不代替断言结果或研究评测。

## 下一步：正式数据接口

以下为后续接入要求，**本次没有实现这些后端接口**。接口应复用[数据契约](DATA_CONTRACT.md)、[实验协议](EXPERIMENT_PROTOCOL.md)与实际 runner 状态，避免在前端创造另一套实验定义。

1. **先接入经过校验的场景清单。** 保留明确的场景与帧 ID、来源文件哈希、图像尺寸、畸变处理、同步信息、`K` 和固定米制 `T_world_camera`。先冻结 `source / validation / test`；预览时生成的时间戳不能自动成为已同步的正式帧记录。
2. **每种数据标明来源。** 传感器观测、模型预测、独立参考和合成演示必须可区分。深度声明光轴 `z_depth` 与米单位、有效掩码、预测模型版本和上下文帧；没有独立几何参考时保持缺失，不能用预测深度填 GT。
3. **读取真实运行与报告。** 先建设只读的 manifest、运行状态及报告适配层，携带配置哈希、代码 / 模型版本、失败状态、耗时与指标来源。只有实际测量才能标成 measured；尚未实现的 runner 不返回虚构进度和结果。
4. **完整保留高斯物理量。** 正式场景接口遵守数据契约的中心位置、`scale_std_m`、`rotation_wxyz`、`color_linear_rgb`、透明度与 `source_observations`。导入器必须明确 log / logit 激活、颜色空间和坐标变换；完成往返与参考渲染核验后，才加入正式 3DGS 导入导出。
5. **在后端满足准入后才开放任务操作。** 许可、环境、数据划分与实际 renderer / optimizer 的验证仍是前置条件。UI 不通过按钮、协议副本或默认相机代替这些条件。任务启动、取消与结果管理属于后续独立集成。

Three.js 按实际使用的成员显式导入，不使用全局 `THREE.*`。前端自己的演示渲染和导出格式也不能自动视为研究渲染器的等价实现。

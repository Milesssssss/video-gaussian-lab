# Video Gaussian Lab · 前端工作台

这个前端用于查看**合成场景演示、仓库中的实验协议状态和本地视频预览**。它保留上传原型的暗色与 cyan 视觉风格，整理为独立的 React / Vite 工程。

**当前没有接通 SHARP、Depth Pro 2、共享高斯优化器或正式研究 runner。** 默认三维场景、深度示意和关联关系来自合成数据；操作滑块产生的变化不构成模型效果或实验结论。正式协议仍为 `execution_enabled=false`。

## 本地运行

使用 **Node.js 24**；本目录的 [`.nvmrc`](.nvmrc) 和 `package.json` 的 `engines` 声明该版本要求。从仓库根目录执行：

```bash
cd ui
npm ci
npm run dev
```

打开 [http://localhost:5173](http://localhost:5173)。开发服务绑定本机地址；端口被占用时会明确失败。这个入口不需要 API key、模型权重、Python worker 或上传服务。

构建产物可单独预览：

```bash
npm run build
npm run preview
```

预览地址为 [http://localhost:4173](http://localhost:4173)。开发服务和构建预览都仅服务前端，不能启动研究实验。

## 可以在界面中做什么

| 入口 | 数据来源与用途 | 解释边界 |
| --- | --- | --- |
| 三维视口、时间轴和参数控制 | 内置合成场景，用于验证布局、交互与可视化表达 | 示例深度、相机和参考轨迹不是真实视频的预测或测量 |
| 特征与深度检查 | 合成数据上的关联及可靠性示意 | 示意分数不是 SHARP / DP2 输出的真实 confidence |
| 实验状态 | 构建所用源码版本的 [`phase1_protocol.json`](../configs/phase1_protocol.json) | 协议快照是只读展示，不是远端运行服务；状态不能由演示滑块更改 |
| 本地视频预览 | 浏览器读取用户选择文件的元数据和抽帧缩略图 | 不估计内参、位姿、深度或独立几何参考，不生成重建场景 |
| 演示点云导出 | 当前合成场景的位置与颜色 | `.synthetic-pointcloud.ply` 是演示点云，不是含完整协方差、旋转和透明度的 3DGS 模型 |

本地视频预览在浏览器中处理文件；本次集成没有视频上传接口或模型请求。可播放格式取决于浏览器及系统的解码器，文件后缀不能保证能够解码。元数据读取或抽帧失败时应显示失败原因，不能用合成数据冒充该视频的处理结果。

抽帧预览仅帮助用户检查视频。它不等于研究协议中的精确帧清单、同步后的相机记录或已冻结的数据划分。

## 开发检查

在 `ui/` 目录执行：

```bash
npm run check
npm test
npm run build
```

| 命令 | 验证范围 |
| --- | --- |
| `npm run check` | TypeScript 类型检查 |
| `npm test` | 合成演示和前端辅助逻辑的单元测试；实际数量以该版本测试输出为准 |
| `npm run build` | 类型检查与 Vite 生产构建 |

[UI CI](../.github/workflows/ui.yml) 在 Node.js 24 下执行 `npm ci` 及上述三项检查，然后安装 Chromium 并执行浏览器 smoke tests。浏览器用例覆盖桌面和手机尺寸的合成工作台、协议状态、键盘探针和本地视频预览；它们不运行真实视频重建、Apple 模型或研究指标评测。

浏览器检查使用生产构建；完成 `npm run build` 后执行：

```bash
npx playwright install --with-deps chromium
npm run test:browser
```

**本地已通过类型检查、16 / 16 项单元测试和生产构建。** Three.js 视口已分为懒加载产物，本次构建为 591.99 kB（gzip 148.45 kB），仍有超过 500 kB 的 chunk 提示；这项提示不代表构建失败。

**浏览器检查尚未通过核验。** 本地 Chromium 在启动阶段受到执行环境 Unix socket 权限限制，桌面与手机尺寸共 6 个用例均未进入页面断言；分支 CI 的执行结果待核验。CI 最后尝试保存 `test-results/**/*.png` 截图，缺少截图不使该保存步骤失败。既有 Python CI 的成功不能代替前端验证；详细记录见[集成说明的验证记录](../docs/UI_INTEGRATION.md#验证记录)。

## 继续接入真实数据前

正式数据入口需要遵守[数据契约](../docs/DATA_CONTRACT.md)和[实验协议](../docs/EXPERIMENT_PROTOCOL.md)：先冻结 source / validation / test，再接入可追溯的内参与米制位姿、预测缓存和真实运行报告。未知字段保持缺失，预测不能充当独立 GT。

前端不得绕过 runner 准入条件，也不能通过显示模型名称来表示模型已经可用。模型执行和产品使用的许可条件另见[第三方记录](../docs/THIRD_PARTY.md)。后续接口、原型来源与裁剪范围见[UI 集成说明](../docs/UI_INTEGRATION.md)。

import { ArrowUpRight } from 'lucide-react';
import { getResearchStatus, phase1Protocol, researchDocuments } from '../lib/researchStatus';

const steps = [
  { id: 'input', title: '真实静态视频与固定相机', status: '待采集与审计', text: '逐帧登记标定 K、米制 T_world_camera、时间同步和图像变换。位姿在所有实验组中固定，并披露其误差来源。' },
  { id: 'split', title: '先划分 source 与留出视角', status: '基础检查已有', text: 'source 用于模型上下文与场景优化；validation / test 只进入独立评测。精确帧 ID 和划分哈希须在推理前登记。' },
  { id: 'initialize', title: '按实验组建立初始化', status: '模型待接入', text: 'S 组使用 SHARP；S2 / S3 增加 DP2 的每关键帧单标量校准；D 组将 DP2 深度反投影成普通高斯。相机平移不随标量缩放。' },
  { id: 'scene', title: '共享场景与多视图优化', status: '光栅化与优化待验证', text: '各组使用同一准入规则和高斯预算。S0 不优化；其余组优化共享 RGB，S3 / D1 另加可见表面深度残差。' },
  { id: 'evaluate', title: '开发冻结后做锁定测试', status: '尚无正式结果', text: '记录成功和失败运行，比较留出渲染、独立几何及完整资源成本。结果可以支持、否定或暂时无法区分初始化策略。' },
];

export function PipelineBlueprint() {
  const status = getResearchStatus();
  const { data_plan: data, depth_context_candidate: context, shared_design: shared } = phase1Protocol;
  const linkClass = 'inline-flex min-h-10 items-center gap-1 text-sm text-cyan-300 underline underline-offset-4 hover:text-cyan-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 rounded-sm';

  return (
    <section aria-labelledby="pipeline-design-title" className="flex-1 min-h-0 overflow-y-auto bg-[#07090E] p-4 sm:p-6">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <header className="max-w-3xl">
          <h1 id="pipeline-design-title" className="text-xl font-semibold text-slate-50">Phase 1 · 固定相机的共享高斯场景</h1>
          <p className="mt-2 text-sm leading-relaxed text-slate-300">先验证 SHARP 初始化与视频深度的实际价值，再决定是否扩展全局特征。以下是仓库研究方案与接入边界。</p>
        </header>

        <div className="flex flex-col gap-2 border-y border-[#1E293B] py-4 text-sm sm:flex-row sm:items-baseline sm:justify-between">
          <p className="text-slate-200">当前实现：契约、几何与实验臂隔离的基础代码及合成测试。</p>
          <span className="shrink-0 text-amber-200">{status.executionEnabled ? '协议允许执行 · UI 尚未连接执行器' : '正式执行未启用'}</span>
        </div>

        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_300px]">
          <section aria-labelledby="pipeline-sequence-title">
            <h2 id="pipeline-sequence-title" className="text-sm font-semibold text-slate-100">研究流程</h2>
            <ol className="mt-3 divide-y divide-[#1E293B] border-y border-[#1E293B]">
              {steps.map((step, index) => (
                <li key={step.id} className="flex items-start gap-3 py-4 sm:gap-4">
                  <span aria-hidden="true" className="pt-0.5 font-mono text-xs tabular-nums text-cyan-400">{String(index + 1).padStart(2, '0')}</span>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                      <h3 className="text-sm font-medium text-slate-100">{step.title}</h3>
                      <span className="text-xs text-slate-400">{step.status}</span>
                    </div>
                    <p className="mt-2 max-w-prose text-sm leading-relaxed text-slate-400">{step.text}</p>
                  </div>
                </li>
              ))}
            </ol>
          </section>

          <aside aria-label="协议中的实验设置" className="self-start rounded-lg border border-[#1E293B] bg-[#0F141D] p-4">
            <h2 className="text-sm font-semibold text-slate-100">待开发冻结的设置</h2>
            <dl className="mt-3 divide-y divide-[#1E293B] text-xs">
              <div className="flex justify-between gap-3 py-3"><dt className="text-slate-400">source 上限</dt><dd className="font-mono tabular-nums text-slate-200">{data.source_keyframes_per_scene_max} 帧 / 场景</dd></div>
              <div className="flex justify-between gap-3 py-3"><dt className="text-slate-400">留出目标</dt><dd className="font-mono tabular-nums text-slate-200">{data.evaluation_views_per_scene_target} 帧 / {data.evaluation_time_blocks_target} 时间块</dd></div>
              <div className="flex justify-between gap-3 py-3"><dt className="text-slate-400">DP2 source 窗口候选</dt><dd className="font-mono tabular-nums text-slate-200">{context.window_size} 帧 / 重叠 {context.overlap}</dd></div>
              <div className="py-3"><dt className="text-slate-400">高斯预算候选</dt><dd className="mt-1.5 font-mono tabular-nums text-slate-200">{shared.capacity_candidates.map((value) => value.toLocaleString()).join(' / ')}</dd></div>
              <div className="flex justify-between gap-3 py-3"><dt className="text-slate-400">模型参数</dt><dd className="text-slate-200">计划冻结，不做微调</dd></div>
            </dl>
            <p className="mt-2 border-t border-[#1E293B] pt-3 text-xs leading-relaxed text-slate-400">主矩阵使用同一预算、同一 source 列表与渲染器。高斯数量相同仍需同时比较完整成本。</p>
          </aside>
        </div>

        <section aria-labelledby="data-boundary-title" className="overflow-hidden rounded-lg border border-[#1E293B] bg-[#0F141D]">
          <h2 id="data-boundary-title" className="border-b border-[#1E293B] px-4 py-3 text-sm font-semibold text-slate-100">图像使用边界</h2>
          <dl className="divide-y divide-[#1E293B] text-sm">
            <div className="grid gap-2 px-4 py-3 sm:grid-cols-[120px_1fr]"><dt className="font-mono text-cyan-300">source</dt><dd className="leading-relaxed text-slate-300">模型输入、DP2 上下文、尺度拟合、准入与共享优化。特征 tracks 只做离线诊断。</dd></div>
            <div className="grid gap-2 px-4 py-3 sm:grid-cols-[120px_1fr]"><dt className="font-mono text-slate-200">validation</dt><dd className="leading-relaxed text-slate-300">开发场景的留出渲染与有限选参，禁止进入模型、匹配或重建优化。</dd></div>
            <div className="grid gap-2 px-4 py-3 sm:grid-cols-[120px_1fr]"><dt className="font-mono text-slate-200">test</dt><dd className="leading-relaxed text-slate-300">方案冻结后评测，结果不回流调参。没有独立几何参考时，几何门槛记为 NA。</dd></div>
          </dl>
        </section>

        <section aria-labelledby="current-boundaries-title" className="grid gap-5 border-y border-[#1E293B] py-5 lg:grid-cols-2">
          <div>
            <h2 id="current-boundaries-title" className="text-sm font-semibold text-slate-100">执行接入仍待完成</h2>
            <p className="mt-2 text-sm leading-relaxed text-slate-400">SHARP / DP2 的正式前向、权重哈希、真实场景数据，以及共享高斯光栅化和优化需要各自的验证记录。此 UI 提供合成示例查看与协议阅读，没有模型启动入口。</p>
            <p className="mt-2 text-xs leading-relaxed text-amber-200">协议允许 G0 = NA，但当前 runner 守卫仍要求独立几何参考。此差异尚未修复，正式执行保持关闭。</p>
          </div>
          <div>
            <h2 className="text-sm font-semibold text-slate-100">研究结论的边界</h2>
            <p className="mt-2 text-sm leading-relaxed text-slate-400">本阶段是离线 batch 重建。视口的帧序列播放不证明在线融合或回访无遗忘；联合位姿优化、学习型全局特征及动态场景留待后续独立研究。</p>
          </div>
        </section>

        <footer className="flex flex-wrap gap-x-5 gap-y-1">
          <a href={researchDocuments.dataContract} target="_blank" rel="noreferrer" className={linkClass}>数据与坐标契约<ArrowUpRight aria-hidden="true" className="h-3.5 w-3.5" /></a>
          <a href={researchDocuments.implementation} target="_blank" rel="noreferrer" className={linkClass}>实现状态<ArrowUpRight aria-hidden="true" className="h-3.5 w-3.5" /></a>
          <a href={researchDocuments.thirdParty} target="_blank" rel="noreferrer" className={linkClass}>模型使用条件<ArrowUpRight aria-hidden="true" className="h-3.5 w-3.5" /></a>
        </footer>
      </div>
    </section>
  );
}

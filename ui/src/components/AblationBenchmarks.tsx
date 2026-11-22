import { ArrowUpRight, Check, Circle } from 'lucide-react';
import { armDescriptions, getResearchStatus, phase1Protocol, researchDocuments } from '../lib/researchStatus';

export function AblationBenchmarks() {
  const status = getResearchStatus();
  const { data_plan: data, run_plan: plan } = phase1Protocol;
  const linkClass = 'inline-flex min-h-10 items-center gap-1 text-sm text-cyan-300 underline underline-offset-4 hover:text-cyan-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 rounded-sm';

  return (
    <section aria-labelledby="research-status-title" className="flex-1 min-h-0 overflow-y-auto bg-[#07090E] p-4 sm:p-6">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <header className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="max-w-3xl">
            <h1 id="research-status-title" className="text-xl font-semibold text-slate-50">实验对照与就绪状态</h1>
            <p className="mt-2 text-sm leading-relaxed text-slate-300">
              Phase 1 研究已知相机的静态视频重建。这里展示仓库中的正式实验登记；视口里的合成示例不计入这些对照。
            </p>
          </div>
          <span className="self-start whitespace-nowrap rounded border border-amber-500/40 px-3 py-1.5 text-xs font-medium text-amber-200">
            {status.executionEnabled ? '配置允许执行 · UI 未连接执行器' : '正式实验未启用'}
          </span>
        </header>

        <dl className="grid grid-cols-2 gap-x-6 gap-y-4 border-y border-[#1E293B] py-4 text-sm sm:grid-cols-4">
          <div><dt className="text-slate-400">协议状态</dt><dd className="mt-1 text-slate-100" data-testid="protocol-status">{status.protocolState === 'draft_not_frozen' ? '草案 · 未冻结' : status.protocolState}</dd></div>
          <div><dt className="text-slate-400">场景计划</dt><dd className="mt-1 font-mono tabular-nums text-slate-100">{data.dev_scenes} dev / {data.locked_test_scenes} test</dd></div>
          <div><dt className="text-slate-400">P1a 计划上限</dt><dd className="mt-1 font-mono tabular-nums text-slate-100">{plan.P1a.planned_run_cap_excluding_logged_failures} 次，含诊断</dd></div>
          <div><dt className="text-slate-400">P1b 优化计划</dt><dd className="mt-1 font-mono tabular-nums text-slate-100">{plan.P1b.optimized_runs} 次 · 尚无运行记录</dd></div>
        </dl>

        <section aria-labelledby="formal-arms-title" className="overflow-hidden rounded-lg border border-[#1E293B] bg-[#0F141D]">
          <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-[#1E293B] px-4 py-3">
            <h2 id="formal-arms-title" className="text-sm font-semibold text-slate-100">正式对照组</h2>
            <p className="text-xs text-slate-400">固定 K / 米制位姿 · 同一高斯预算 · 冻结模型</p>
          </div>
          <div className="overflow-x-auto" tabIndex={0} aria-label="六个正式实验组，可横向滚动" role="region">
            <table className="w-full min-w-[650px] text-left text-xs sm:text-sm">
              <caption className="sr-only">Phase 1 六组定义，数据来自仓库协议；当前没有各组运行结果。</caption>
              <thead className="bg-[#0B0E17] text-slate-400">
                <tr>
                  <th scope="col" className="px-4 py-3">实验组 / 用途</th>
                  <th scope="col" className="px-4 py-3">初始化</th>
                  <th scope="col" className="px-4 py-3">DP2 标量校准</th>
                  <th scope="col" className="px-4 py-3">共享 RGB 优化</th>
                  <th scope="col" className="px-4 py-3">深度残差</th>
                  <th scope="col" className="px-4 py-3">运行结果</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1E293B] text-slate-200">
                {phase1Protocol.arms.map((arm) => (
                  <tr key={arm.id}>
                    <th scope="row" className="px-4 py-3 font-normal">
                      <span className="font-mono font-semibold text-cyan-300">{arm.id}</span>
                      <span className="mt-1 block text-xs text-slate-400">{armDescriptions[arm.id] ?? arm.id}</span>
                    </th>
                    <td className="px-4 py-3">{arm.initializer === 'sharp' ? 'SHARP' : 'DP2 深度反投影'}</td>
                    <td className="px-4 py-3">{arm.depth_scale ? '每帧一个标量' : '不使用'}</td>
                    <td className="px-4 py-3">{arm.rgb_optimization ? '计划启用' : '不优化'}</td>
                    <td className="px-4 py-3">{arm.depth_loss ? '计划启用' : '不使用'}</td>
                    <td className="px-4 py-3 text-slate-400">未提供</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="border-t border-[#1E293B] px-4 py-3 text-xs leading-relaxed text-slate-400">
            S1 / S2 区分尺度初始化，S2 / S3 和 D0 / D1 区分深度残差；S2 / D0、S3 / D1 比较整套初始化策略。S0 只用于诊断。
          </p>
        </section>

        <div className="grid gap-6 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
          <section aria-labelledby="entry-conditions-title">
            <h2 id="entry-conditions-title" className="text-sm font-semibold text-slate-100">入口条件登记</h2>
            <ul className="mt-3 divide-y divide-[#1E293B] border-y border-[#1E293B]">
              {status.conditions.map((condition) => (
                <li key={condition.key} className="flex items-start gap-3 py-3">
                  {condition.met ? <Check aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" /> : <Circle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" />}
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap justify-between gap-x-3 gap-y-1 text-xs">
                      <span className="font-medium text-slate-200">{condition.label}</span>
                      <span className={condition.met ? 'text-emerald-300' : 'text-amber-200'}>
                        {condition.met ? '已登记满足' : condition.requiredByProtocol ? '待完成' : 'G0 · NA / 守卫待满足'}
                      </span>
                    </div>
                    <p className="mt-1 text-xs leading-relaxed text-slate-400">{condition.detail}</p>
                  </div>
                </li>
              ))}
            </ul>
          </section>

          <section aria-labelledby="results-title" className="self-start rounded-lg border border-[#1E293B] bg-[#0F141D] p-4">
            <h2 id="results-title" className="text-sm font-semibold text-slate-100">实测指标</h2>
            <p data-testid="research-results-state" className="mt-3 text-base font-medium text-slate-200">
              {status.resultState === 'not_provided' ? '尚未提供实验结果' : '结果登记已存在，展示接口待接入'}
            </p>
            <p className="mt-2 text-sm leading-relaxed text-slate-400">完成真实场景运行并登记结果后，才可比较各组质量与成本。</p>
            <dl className="mt-4 divide-y divide-[#1E293B] text-xs">
              <div className="flex flex-wrap justify-between gap-2 py-3"><dt className="text-slate-300">PSNR / SSIM / LPIPS</dt><dd className="text-slate-400">— 未提供</dd></div>
              <div className="flex flex-wrap justify-between gap-2 py-3"><dt className="text-slate-300">独立几何误差</dt><dd className="text-slate-400">{status.geometryState === 'NA' ? 'NA · 缺少独立参考' : '— 待评测'}</dd></div>
              <div className="flex flex-wrap justify-between gap-2 py-3"><dt className="text-slate-300">完整耗时 / 峰值显存</dt><dd className="text-slate-400">— 未提供</dd></div>
            </dl>
            <p className="mt-3 border-t border-[#1E293B] pt-3 text-xs leading-relaxed text-slate-400">
              合成示例不会填入本表。协议允许缺少独立参考时将 G0 记为 NA，但当前 runner 守卫仍要求独立参考；正式执行继续关闭。缺失结果不按零误差处理。
            </p>
          </section>
        </div>

        <footer className="flex flex-wrap items-center gap-x-5 gap-y-1 border-t border-[#1E293B] pt-3">
          <a href={researchDocuments.protocol} target="_blank" rel="noreferrer" className={linkClass}>实验协议<ArrowUpRight aria-hidden="true" className="h-3.5 w-3.5" /></a>
          <a href={researchDocuments.registration} target="_blank" rel="noreferrer" className={linkClass}>状态登记来源<ArrowUpRight aria-hidden="true" className="h-3.5 w-3.5" /></a>
          <p className="w-full text-xs leading-relaxed text-slate-400">本页读取随当前 UI 版本提供的仓库登记。实时任务与结果服务尚未接入。</p>
        </footer>
      </div>
    </section>
  );
}

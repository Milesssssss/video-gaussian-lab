import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import type { FrameData } from '../types/pipeline';

interface FeatureDepthInspectorProps {
  frames: FrameData[];
  activeFrameIndex: number;
  onSelectFrame: (idx: number) => void;
}

type Probe = { frame: FrameData; u: number; v: number };

export function FeatureDepthInspector({ frames, activeFrameIndex, onSelectFrame }: FeatureDepthInspectorProps) {
  const rgbCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const depthCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [probe, setProbe] = useState<Probe | null>(null);
  const selectedIndex = Math.max(0, Math.min(activeFrameIndex, frames.length - 1));
  const frame = frames[selectedIndex];
  const hasForwardMatches = Boolean(frame && selectedIndex + 1 < frames.length && frame.matchesToNext.length);
  const hasPreviousMatches = Boolean(selectedIndex > 0 && frames[selectedIndex - 1]?.matchesToNext.length);
  const pairSource = hasForwardMatches ? frame : hasPreviousMatches ? frames[selectedIndex - 1] : frame;
  const pairTarget = hasForwardMatches ? frames[selectedIndex + 1] : hasPreviousMatches ? frame : null;
  const displayMatches = pairTarget ? pairSource?.matchesToNext ?? [] : [];
  const activeProbe = probe?.frame === frame ? probe : null;

  useEffect(() => {
    setProbe(null);
  }, [frame]);

  useEffect(() => {
    const canvas = rgbCanvasRef.current;
    if (!canvas || !pairSource) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const image = ctx.createImageData(pairSource.width, pairSource.height);
    image.data.set(pairSource.rgbPixels);
    ctx.putImageData(image, 0, 0);
    ctx.lineWidth = Math.max(0.6, pairSource.width / 320);
    for (const match of displayMatches) {
      ctx.strokeStyle = match.inlier ? '#22d3ee' : '#f59e0b';
      ctx.fillStyle = ctx.strokeStyle;
      ctx.beginPath();
      ctx.moveTo(match.u1, match.v1);
      ctx.lineTo(match.u2, match.v2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(match.u1, match.v1, Math.max(1.1, pairSource.width / 180), 0, Math.PI * 2);
      ctx.fill();
    }
  }, [pairSource, pairTarget]);

  useEffect(() => {
    const canvas = depthCanvasRef.current;
    if (!canvas || !frame) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const image = ctx.createImageData(frame.width, frame.height);
    const span = Math.max(0.001, frame.maxDepth - frame.minDepth);
    for (let i = 0; i < frame.depthMapMeters.length; i++) {
      const depth = frame.depthMapMeters[i];
      if (!Number.isFinite(depth) || depth <= 0) {
        image.data.set([7, 9, 14, 255], i * 4);
        continue;
      }
      const t = 1 - Math.max(0, Math.min(1, (depth - frame.minDepth) / span));
      image.data[i * 4] = Math.min(255, Math.round(t > 0.55 ? 255 * (t - 0.55) * 2.2 : 12));
      image.data[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(45 + 195 * Math.sin(t * Math.PI * 0.85))));
      image.data[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(70 + 175 * (1 - Math.abs(t - 0.45) * 1.6))));
      image.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }, [frame]);

  if (!frame || !pairSource) {
    return <section className="flex-1 overflow-y-auto p-6 text-sm text-slate-300"><h1 className="text-lg font-semibold text-slate-100">特征与深度示例</h1><p className="mt-3">暂无可检查帧，请先选择一个合成场景。</p></section>;
  }

  const setProbeAt = (u: number, v: number) => {
    setProbe({
      frame,
      u: Math.max(0, Math.min(frame.width - 1, Math.round(u))),
      v: Math.max(0, Math.min(frame.height - 1, Math.round(v))),
    });
  };

  const handlePointer = (event: PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    setProbeAt(
      Math.floor(((event.clientX - rect.left) / rect.width) * frame.width),
      Math.floor(((event.clientY - rect.top) / rect.height) * frame.height),
    );
  };

  const handleProbeKeyDown = (event: KeyboardEvent<HTMLCanvasElement>) => {
    const origin = activeProbe ?? { u: Math.floor(frame.width / 2), v: Math.floor(frame.height / 2) };
    const step = event.shiftKey ? 10 : 1;
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step],
    };
    if (moves[event.key]) {
      event.preventDefault();
      const [du, dv] = moves[event.key];
      setProbeAt(origin.u + du, origin.v + dv);
    } else if (event.key === 'Home' || event.key === 'End') {
      event.preventDefault();
      setProbeAt(event.key === 'Home' ? 0 : frame.width - 1, event.key === 'Home' ? 0 : frame.height - 1);
    } else if (event.key === 'Escape') {
      event.preventDefault();
      setProbe(null);
    }
  };

  const probeDepth = activeProbe ? frame.depthMapMeters[activeProbe.v * frame.width + activeProbe.u] : null;
  const validProbeDepth = probeDepth !== null && Number.isFinite(probeDepth) && probeDepth > 0;
  const inliers = displayMatches.filter((match) => match.inlier).length;
  const matchLabel = pairTarget
    ? 'F' + (pairSource.frameIndex + 1) + ' → F' + (pairTarget.frameIndex + 1)
    : '暂无相邻匹配';
  const focusClass = 'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:ring-offset-2 focus-visible:ring-offset-[#0F141D]';

  return (
    <section aria-labelledby="feature-depth-title" className="flex-1 min-h-0 overflow-y-auto bg-[#07090E] p-4 sm:p-6">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <header className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
          <div className="max-w-3xl">
            <h1 id="feature-depth-title" className="text-xl font-semibold text-slate-50">合成帧 · 特征与深度检查</h1>
            <p className="mt-2 text-sm leading-relaxed text-slate-300">RGB、深度和匹配均由示例场景生成，用于检查界面与坐标交互。这里没有 SHARP / DP2 推理输出或真实采集测量。</p>
          </div>
          <label className="flex shrink-0 flex-col gap-1.5 text-xs text-slate-400" htmlFor="frame-select-inspector">
            当前检查帧
            <select
              id="frame-select-inspector"
              value={selectedIndex}
              onChange={(event) => onSelectFrame(Number(event.target.value))}
              className={'min-h-11 rounded-md border border-[#1E293B] bg-[#0F141D] px-3 text-sm text-slate-100 ' + focusClass}
            >
              {frames.map((item, index) => <option key={item.frameIndex} value={index}>{'F' + String(item.frameIndex + 1).padStart(2, '0') + ' · ' + item.timestampSec.toFixed(2) + ' s'}</option>)}
            </select>
          </label>
        </header>

        <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
          <figure className="flex min-w-0 flex-col gap-3 rounded-lg border border-[#1E293B] bg-[#0F141D] p-4">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h2 className="text-sm font-semibold text-slate-100">合成 RGB 与匹配连线</h2>
              <span className="font-mono text-xs tabular-nums text-cyan-300">{matchLabel}</span>
            </div>
            <canvas
              ref={rgbCanvasRef}
              width={pairSource.width}
              height={pairSource.height}
              role="img"
              aria-label={'合成源帧 F' + (pairSource.frameIndex + 1) + '，显示 ' + displayMatches.length + ' 条示例匹配'}
              className="block h-auto w-full rounded border border-[#1E293B] bg-[#07090E]"
              style={{ aspectRatio: pairSource.width + ' / ' + pairSource.height }}
            >匹配坐标可在下方表格中查看。</canvas>
            <figcaption className="flex flex-col gap-2 text-xs leading-relaxed text-slate-400">
              <p>图像为匹配起点 F{pairSource.frameIndex + 1}。连线将相邻帧坐标投到同一画布，仅用于观察位移。</p>
              <p><span className="text-cyan-300">青色：示例内点 {inliers}</span><span className="mx-2">/</span><span className="text-amber-300">琥珀：示例外点 {displayMatches.length - inliers}</span></p>
            </figcaption>
          </figure>

          <figure className="flex min-w-0 flex-col gap-3 rounded-lg border border-[#1E293B] bg-[#0F141D] p-4">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h2 className="text-sm font-semibold text-slate-100">合成深度探针</h2>
              <span className="font-mono text-xs tabular-nums text-slate-300">{'F' + (frame.frameIndex + 1) + ' · 示例单位 m'}</span>
            </div>
            <div className="relative">
              <canvas
                ref={depthCanvasRef}
                width={frame.width}
                height={frame.height}
                tabIndex={0}
                role="group"
                aria-label="合成深度探针，使用方向键移动"
                aria-describedby="depth-probe-help depth-probe-readout"
                onFocus={() => { if (!activeProbe) setProbeAt(Math.floor(frame.width / 2), Math.floor(frame.height / 2)); }}
                onKeyDown={handleProbeKeyDown}
                onPointerDown={handlePointer}
                onPointerMove={(event) => { if (event.pointerType === 'mouse') handlePointer(event); }}
                className={'block h-auto w-full cursor-crosshair rounded border border-[#1E293B] bg-[#07090E] ' + focusClass}
                style={{ aspectRatio: frame.width + ' / ' + frame.height }}
              >按 Tab 聚焦，方向键选择像素，读数显示在下方。</canvas>
              {activeProbe && (
                <span
                  aria-hidden="true"
                  className="pointer-events-none absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-black/40"
                  style={{ left: ((activeProbe.u + 0.5) / frame.width) * 100 + '%', top: ((activeProbe.v + 0.5) / frame.height) * 100 + '%' }}
                />
              )}
            </div>
            <figcaption className="flex flex-col gap-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <output id="depth-probe-readout" aria-live="polite" aria-atomic="true" className="font-mono text-sm tabular-nums text-cyan-300" data-testid="depth-probe-readout">
                  {activeProbe ? '(' + activeProbe.u + ', ' + activeProbe.v + ') · ' + (validProbeDepth ? probeDepth.toFixed(3) + ' m（合成）' : '无效深度') : '尚未选择像素'}
                </output>
                <button type="button" onClick={() => setProbe(null)} disabled={!activeProbe} className={'min-h-10 rounded px-2 text-xs text-slate-300 hover:bg-[#1E293B] disabled:opacity-50 ' + focusClass}>清除探针</button>
              </div>
              <p id="depth-probe-help" className="text-xs leading-relaxed text-slate-400">移动鼠标或轻触图像；键盘聚焦后用方向键移动 1 px，Shift + 方向键移动 10 px，Home / End 跳至两角，Esc 清除。</p>
              <p className="font-mono text-xs tabular-nums text-slate-400">{'示例区间 ' + frame.minDepth.toFixed(2) + '–' + frame.maxDepth.toFixed(2) + ' m · 示例焦距 ' + frame.estimatedFocalPx.toFixed(0) + ' px'}</p>
            </figcaption>
          </figure>
        </div>

        <section aria-labelledby="synthetic-matches-title" className="overflow-hidden rounded-lg border border-[#1E293B] bg-[#0F141D]">
          <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-[#1E293B] px-4 py-3">
            <h2 id="synthetic-matches-title" className="text-sm font-semibold text-slate-100">合成匹配坐标</h2>
            <p className="text-xs text-slate-400">展示前 {Math.min(12, displayMatches.length)} / {displayMatches.length} 条 · {matchLabel}</p>
          </div>
          {displayMatches.length ? (
            <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="合成匹配坐标表，可横向滚动">
              <table className="w-full min-w-[480px] text-left text-xs">
                <caption className="sr-only">匹配来自合成场景，残差和内点标签不构成真实实验结果。</caption>
                <thead className="bg-[#0B0E17] text-slate-400"><tr><th scope="col" className="px-4 py-3">特征 ID</th><th scope="col" className="px-4 py-3">起点 (u, v)</th><th scope="col" className="px-4 py-3">终点 (u, v)</th><th scope="col" className="px-4 py-3">示例残差 px</th><th scope="col" className="px-4 py-3">示例标签</th></tr></thead>
                <tbody className="divide-y divide-[#1E293B] font-mono tabular-nums text-slate-200">
                  {displayMatches.slice(0, 12).map((match) => <tr key={match.id}><th scope="row" className="px-4 py-3 font-normal">{match.id}</th><td className="px-4 py-3">{match.u1.toFixed(1) + ', ' + match.v1.toFixed(1)}</td><td className="px-4 py-3">{match.u2.toFixed(1) + ', ' + match.v2.toFixed(1)}</td><td className="px-4 py-3">{match.reprojErrorPx.toFixed(2)}</td><td className={'px-4 py-3 font-sans ' + (match.inlier ? 'text-cyan-300' : 'text-amber-300')}>{match.inlier ? '内点' : '外点'}</td></tr>)}
                </tbody>
              </table>
            </div>
          ) : <p className="px-4 py-5 text-sm text-slate-400">当前帧没有可展示的相邻匹配。</p>}
        </section>
      </div>
    </section>
  );
}

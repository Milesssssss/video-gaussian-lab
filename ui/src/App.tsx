import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import { AblationBenchmarks } from './components/AblationBenchmarks';
import { ControlSidebar } from './components/ControlSidebar';
import { FeatureDepthInspector } from './components/FeatureDepthInspector';
import { PipelineBlueprint } from './components/PipelineBlueprint';
import { TimelineDock } from './components/TimelineDock';
import { TopBar } from './components/TopBar';
import { VideoImportDialog } from './components/VideoImportDialog';
import { ErrorBoundary } from './components/ErrorBoundary';
import { generatePresetVideoSequence, runAggregationPipeline } from './demo/gaussianAggregation';
import { computeDemoDepthConfidenceMap } from './demo/depthConfidence';
import { downloadText, exportDemoPointCloud } from './lib/exportDemo';
import type { VideoPreview } from './lib/videoPreview';
import type { ActiveTab, ColorMode, PipelineParams, VideoPresetId } from './types/pipeline';

const GaussianViewport = lazy(() => import('./components/GaussianViewport').then((module) => ({ default: module.GaussianViewport })));

const INITIAL_LAYERS = {
  showSplats: true, showCameras: true, showTracks: false,
  showFeatureLinks: true, showRawGhost: true, showDepthConfidence: false,
};

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('viewport');
  const [selectedPreset, setSelectedPreset] = useState<Exclude<VideoPresetId, 'custom'>>('courtyard');
  const [controlsOpen, setControlsOpen] = useState(false);
  const [videoOpen, setVideoOpen] = useState(false);
  const [videoPreview, setVideoPreview] = useState<VideoPreview | null>(null);
  const [params, setParams] = useState<PipelineParams>({
    mode: 'global_fused', depthTemporalWeight: 0.9,
    globalConstraintWeight: 0.9, mergeRadiusMeters: 0.1, maxActiveFrame: 15,
  });
  const [colorMode, setColorMode] = useState<ColorMode>('rgb');
  const [layers, setLayers] = useState(INITIAL_LAYERS);
  const [isPlaying, setIsPlaying] = useState(false);
  const [notice, setNotice] = useState('');

  const sequence = useMemo(() => generatePresetVideoSequence(selectedPreset), [selectedPreset]);
  const activeFrame = Math.min(params.maxActiveFrame, sequence.frames.length - 1);
  const result = useMemo(() => runAggregationPipeline(sequence, { ...params, maxActiveFrame: activeFrame }), [sequence, params, activeFrame]);
  const depthWeight = params.mode === 'raw_sharp' ? 0 : params.depthTemporalWeight;
  const depthConfidence = useMemo(() => computeDemoDepthConfidenceMap(
    sequence.frames[activeFrame], activeFrame > 0 ? sequence.frames[activeFrame - 1] : null, depthWeight,
  ), [sequence, activeFrame, depthWeight]);

  useEffect(() => {
    if (!isPlaying) return;
    if (activeFrame >= sequence.frames.length - 1) { setIsPlaying(false); return; }
    const timer = window.setTimeout(() => setParams((previous) => ({ ...previous, maxActiveFrame: activeFrame + 1 })), 600);
    return () => window.clearTimeout(timer);
  }, [isPlaying, activeFrame, sequence.frames.length]);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(''), 5000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const selectFrame = (frame: number) => {
    setIsPlaying(false);
    setParams((previous) => ({ ...previous, maxActiveFrame: Math.max(0, Math.min(frame, sequence.frames.length - 1)) }));
  };

  return (
    <div className="app-shell" data-pipeline-mode={params.mode} data-active-tab={activeTab} data-sequence-id={sequence.id}>
      <a href="#workspace" className="skip-link">跳到工作区</a>
      <TopBar activeTab={activeTab} controlsOpen={controlsOpen} hasVideoPreview={Boolean(videoPreview)}
        onToggleControls={() => setControlsOpen((open) => !open)}
        onSelectTab={(tab) => { setActiveTab(tab); setIsPlaying(false); }}
        onOpenVideoModal={() => setVideoOpen(true)} />
      <div className="source-strip">
        <span className="source-badge">合成演示</span>
        <p>预设场景用于交互验证。SHARP 与 Depth Pro 2 推理、正式重建结果尚未接入。</p>
        <button type="button" onClick={() => { setActiveTab('benchmarks'); setIsPlaying(false); }} className="text-cyan-300 shrink-0 hover:text-cyan-100">查看实验状态</button>
      </div>
      <main id="workspace" tabIndex={-1} className="workspace">
        {activeTab === 'viewport' && (
          <ControlSidebar sequence={sequence} open={controlsOpen}
            onSelectPreset={(preset) => { setIsPlaying(false); setSelectedPreset(preset); setParams((p) => ({ ...p, maxActiveFrame: 15 })); }}
            params={{ ...params, maxActiveFrame: activeFrame }} onUpdateParams={(patch) => setParams((p) => ({ ...p, ...patch }))}
            colorMode={colorMode} onChangeColorMode={setColorMode} layers={layers}
            onToggleLayer={(key) => setLayers((previous) => ({ ...previous, [key]: !previous[key] }))}
            depthConfidenceMean={depthConfidence.meanConfidence} result={result}
            onExportPly={() => {
              downloadText(exportDemoPointCloud(sequence, result.splats), `${selectedPreset}.synthetic-pointcloud.ply`);
              setNotice('已导出演示点云，仅包含位置与颜色。');
            }} />
        )}
        <div className="workspace-content">
          <ErrorBoundary key={activeTab}>
            {activeTab === 'viewport' && (
              <section className="viewport-stage" aria-label="合成场景三维预览">
                <Suspense fallback={<div className="view-loading" role="status">正在加载三维预览…</div>}>
                  <GaussianViewport result={result} gtPoses={sequence.frames.map((frame) => frame.gtPose)}
                    rawPoses={sequence.frames.map((frame) => frame.rawPose)} activeFrameIndex={activeFrame}
                    colorMode={colorMode} layers={layers} depthConfidence={depthConfidence} depthTemporalWeight={depthWeight} />
                </Suspense>
                <TimelineDock frames={sequence.frames} activeFrameIndex={activeFrame} isPlaying={isPlaying}
                  onTogglePlay={() => {
                    if (!isPlaying && activeFrame === sequence.frames.length - 1) selectFrame(0);
                    setIsPlaying((playing) => !playing);
                  }} onSelectFrame={selectFrame} />
              </section>
            )}
            {activeTab === 'inspector' && <FeatureDepthInspector frames={sequence.frames} activeFrameIndex={activeFrame} onSelectFrame={selectFrame} />}
            {activeTab === 'benchmarks' && <AblationBenchmarks />}
            {activeTab === 'blueprint' && <PipelineBlueprint />}
          </ErrorBoundary>
        </div>
      </main>
      <footer className="status-bar">
        <span><span className="status-dot" />Phase 1 · UI 原型</span>
        <span className="truncate">{videoPreview ? `本地视频已预览：${videoPreview.name} · 未生成场景` : '预设数据在本地计算 · 无模型请求'}</span>
      </footer>
      <div role="status" aria-live="polite" className={notice ? 'toast' : 'sr-only'}>{notice}</div>
      <VideoImportDialog open={videoOpen} preview={videoPreview} onPreviewReady={setVideoPreview} onClose={() => setVideoOpen(false)} />
    </div>
  );
}

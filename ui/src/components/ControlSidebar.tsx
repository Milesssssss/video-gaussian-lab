import { Download } from 'lucide-react';
import type { AggregationResult, ColorMode, PipelineMode, PipelineParams, VideoPresetId, VideoSequence } from '../types/pipeline';

type Layers = { showSplats: boolean; showCameras: boolean; showTracks: boolean; showFeatureLinks: boolean; showRawGhost: boolean; showDepthConfidence: boolean };
interface Props {
  sequence: VideoSequence; open: boolean;
  onSelectPreset: (preset: Exclude<VideoPresetId, 'custom'>) => void;
  params: PipelineParams; onUpdateParams: (patch: Partial<PipelineParams>) => void;
  colorMode: ColorMode; onChangeColorMode: (mode: ColorMode) => void;
  layers: Layers; onToggleLayer: (key: keyof Layers) => void;
  depthConfidenceMean: number; onExportPly: () => void; result: AggregationResult;
}
const PRESETS = [{ id: 'courtyard', label: '建筑庭院环绕' }, { id: 'workbench', label: '实验桌面推轨' }, { id: 'greenhouse', label: '植物温室穿行' }] as const;
const MODES: { id: PipelineMode; label: string; description: string }[] = [
  { id: 'raw_sharp', label: '逐帧', description: '展示人为加入的尺度扰动与轨迹偏移。' },
  { id: 'depth_pro_2l', label: '深度校正', description: '演示深度权重对合成点分布的影响。' },
  { id: 'global_fused', label: '聚合', description: '使用合成参考轨迹做插值与邻域合并示意。' },
];
const LAYERS: { key: keyof Layers; label: string }[] = [
  { key: 'showSplats', label: '高斯形状预览' }, { key: 'showCameras', label: '相机与视锥' },
  { key: 'showFeatureLinks', label: '跨帧匹配连线' }, { key: 'showTracks', label: '特征射线' },
  { key: 'showRawGhost', label: '扰动轨迹对比' }, { key: 'showDepthConfidence', label: '启发式深度热图' },
];

export function ControlSidebar({ sequence, open, onSelectPreset, params, onUpdateParams, colorMode, onChangeColorMode, layers, onToggleLayer, depthConfidenceMean, onExportPly, result }: Props) {
  const { metrics } = result;
  return (
    <aside id="scene-controls" aria-label="演示参数与图层" className={`control-sidebar ${open ? 'is-open' : ''}`}>
      <section className="control-section">
        <div className="section-heading"><h2>演示场景</h2><span className="numeric text-slate-400">{sequence.frames.length} 帧</span></div>
        <div className="flex flex-col gap-2">{PRESETS.map((preset) => <button type="button" key={preset.id} aria-pressed={sequence.id === preset.id} className={`preset-button ${sequence.id === preset.id ? 'is-active' : ''}`} onClick={() => onSelectPreset(preset.id)}>{preset.label}</button>)}</div>
        <p className="helper-copy">{sequence.subtitle}</p>
      </section>
      <section className="control-section">
        <div className="section-heading"><h2>聚合示意</h2><span className="text-slate-400">演示参数</span></div>
        <div className="segmented-control" role="group" aria-label="聚合演示模式">{MODES.map((mode) => <button key={mode.id} type="button" aria-pressed={params.mode === mode.id} onClick={() => onUpdateParams({ mode: mode.id })}>{mode.label}</button>)}</div>
        <p className="helper-copy">{MODES.find((mode) => mode.id === params.mode)?.description}</p>
        <label className="slider-field"><span>深度校正权重<output>{params.mode === 'raw_sharp' ? '0.00' : params.depthTemporalWeight.toFixed(2)}</output></span>
          <input className="lab-slider" type="range" aria-label="深度校正权重" min={0} max={1} step={0.05} disabled={params.mode === 'raw_sharp'} value={params.depthTemporalWeight} onChange={(event) => onUpdateParams({ depthTemporalWeight: Number(event.target.value) })} />
        </label>
        <label className="slider-field"><span>参考轨迹插值<output>{params.mode === 'global_fused' ? params.globalConstraintWeight.toFixed(2) : '0.00'}</output></span>
          <input className="lab-slider" type="range" aria-label="参考轨迹插值" min={0} max={1} step={0.05} disabled={params.mode !== 'global_fused'} value={params.globalConstraintWeight} onChange={(event) => onUpdateParams({ globalConstraintWeight: Number(event.target.value) })} />
        </label>
        <label className="slider-field"><span>邻域合并半径<output>{params.mergeRadiusMeters.toFixed(2)} m</output></span>
          <input className="lab-slider" type="range" aria-label="邻域合并半径" min={0.02} max={0.30} step={0.02} disabled={params.mode !== 'global_fused'} value={params.mergeRadiusMeters} onChange={(event) => onUpdateParams({ mergeRadiusMeters: Number(event.target.value) })} />
        </label>
      </section>
      <section className="control-section">
        <div className="section-heading"><h2>视口与图层</h2></div>
        <div className="segmented-control" role="group" aria-label="视口着色">{([{ id: 'rgb', label: 'RGB' }, { id: 'depth', label: '合成深度' }, { id: 'error', label: '演示残差' }] as const).map((mode) => <button key={mode.id} type="button" aria-pressed={colorMode === mode.id} onClick={() => onChangeColorMode(mode.id)}>{mode.label}</button>)}</div>
        <div className="layer-list">{LAYERS.map((layer) => <label key={layer.key}><input type="checkbox" checked={layers[layer.key]} onChange={() => onToggleLayer(layer.key)} /><span>{layer.label}</span>{layer.key === 'showDepthConfidence' && <span className="numeric text-cyan-300 ml-auto">{Math.round(depthConfidenceMean * 100)}%</span>}</label>)}</div>
        <p className="helper-copy">热图是演示评分，未经置信度校准。</p>
      </section>
      <section className="control-section" aria-label="合成场景统计">
        <div className="section-heading"><h2>当前演示统计</h2></div>
        <dl className="stat-list">
          <div><dt>纳入帧数</dt><dd>{metrics.activeFrames} / {sequence.frames.length}</dd></div>
          <div><dt>输入基元</dt><dd>{metrics.rawSplatCount.toLocaleString()}</dd></div>
          <div><dt>输出基元</dt><dd data-testid="metric-splats">{metrics.outputSplatCount.toLocaleString()}</dd></div>
          <div><dt>合成参考位置差</dt><dd>{metrics.meanAteMm.toFixed(1)} <span>mm</span></dd></div>
        </dl>
        <button type="button" className="secondary-button w-full" onClick={onExportPly}><Download size={15} aria-hidden="true" />导出演示点云</button>
        <p className="helper-copy">PLY 仅含位置与颜色，不是训练后的高斯场景。</p>
      </section>
    </aside>
  );
}

import { useEffect, useRef, useState } from 'react';
import { ChevronDown, ChevronUp, Pause, Play, SkipBack, SkipForward } from 'lucide-react';
import type { FrameData } from '../types/pipeline';

interface Props { frames: FrameData[]; activeFrameIndex: number; isPlaying: boolean; onTogglePlay: () => void; onSelectFrame: (index: number) => void }
export function TimelineDock({ frames, activeFrameIndex, isPlaying, onTogglePlay, onSelectFrame }: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frame = frames[activeFrameIndex];
  useEffect(() => {
    if (!frame || collapsed) return;
    const context = canvasRef.current?.getContext('2d');
    if (context) {
      const image = context.createImageData(frame.width, frame.height);
      image.data.set(frame.rgbPixels);
      context.putImageData(image, 0, 0);
    }
  }, [frame, collapsed]);
  if (!frame) return null;
  return (
    <div className="timeline-dock">
      <button type="button" className="timeline-handle" onClick={() => setCollapsed((value) => !value)} aria-expanded={!collapsed} aria-controls="frame-timeline">
        {collapsed ? <ChevronUp size={14} aria-hidden="true" /> : <ChevronDown size={14} aria-hidden="true" />}{collapsed ? '展开时间轴' : '收起时间轴'}
      </button>
      {!collapsed && <div id="frame-timeline" className="timeline-panel">
        <div className="playback-controls">
          <button type="button" className="icon-button" aria-label="上一帧" disabled={activeFrameIndex === 0} onClick={() => onSelectFrame(activeFrameIndex - 1)}><SkipBack size={16} aria-hidden="true" /></button>
          <button type="button" className="primary-button" onClick={onTogglePlay}>{isPlaying ? <Pause size={15} aria-hidden="true" /> : <Play size={15} aria-hidden="true" />}{isPlaying ? '暂停演示' : '播放演示'}</button>
          <button type="button" className="icon-button" aria-label="下一帧" disabled={activeFrameIndex === frames.length - 1} onClick={() => onSelectFrame(activeFrameIndex + 1)}><SkipForward size={16} aria-hidden="true" /></button>
        </div>
        <div className="frame-scrubber">
          <div className="frame-caption"><span>帧 <strong className="text-slate-100">{String(activeFrameIndex + 1).padStart(2, '0')}</strong> / {frames.length}<span className="ml-3">{frame.timestampSec.toFixed(2)} s</span></span><span>合成序列</span></div>
          <input type="range" aria-label="当前演示帧" min={0} max={frames.length - 1} step={1} value={activeFrameIndex} onChange={(event) => onSelectFrame(Number(event.target.value))} className="sr-only focus:not-sr-only lab-slider" />
          <div className="frame-markers" role="group" aria-label="选择演示帧">{frames.map((item, index) => <button key={item.frameIndex} type="button" aria-label={`跳转到第 ${index + 1} 帧`} aria-pressed={index === activeFrameIndex} className={`${index < activeFrameIndex ? 'is-included' : ''} ${index === activeFrameIndex ? 'is-current' : ''}`} onClick={() => onSelectFrame(index)} />)}</div>
        </div>
        <figure className="timeline-thumbnail"><canvas ref={canvasRef} width={frame.width} height={frame.height} aria-label={`第 ${activeFrameIndex + 1} 帧合成 RGB 缩略图`} /><figcaption>合成 RGB</figcaption></figure>
      </div>}
    </div>
  );
}

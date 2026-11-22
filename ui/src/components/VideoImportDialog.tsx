import { useEffect, useRef, useState } from 'react';
import { Film, X } from 'lucide-react';
import { readVideoPreview, type VideoPreview } from '../lib/videoPreview';

interface Props { open: boolean; preview: VideoPreview | null; onPreviewReady: (preview: VideoPreview) => void; onClose: () => void }
export function VideoImportDialog({ open, preview, onPreviewReady, onClose }: Props) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const [busy, setBusy] = useState(false);
  const [completed, setCompleted] = useState(0);
  const [error, setError] = useState('');
  const [filename, setFilename] = useState('');

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open) { controllerRef.current?.abort(); dialog.close(); setBusy(false); }
    return () => { controllerRef.current?.abort(); };
  }, [open]);

  const selectFile = async (file?: File) => {
    if (!file) return;
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setError(''); setCompleted(0); setBusy(true); setFilename(file.name);
    try {
      const result = await readVideoPreview(file, controller.signal, (count) => {
        if (controllerRef.current === controller && !controller.signal.aborted) setCompleted(count);
      });
      if (!controller.signal.aborted && controllerRef.current === controller) onPreviewReady(result);
    } catch (reason) {
      if (!controller.signal.aborted && controllerRef.current === controller) setError(reason instanceof Error ? reason.message : '视频读取失败，请重新选择文件。');
    } finally {
      if (controllerRef.current === controller) setBusy(false);
    }
  };

  return (
    <dialog ref={dialogRef} className="video-dialog" aria-labelledby="video-dialog-title" aria-describedby="video-dialog-description" onCancel={(event) => { event.preventDefault(); onClose(); }}>
      <div className="dialog-header"><div><h2 id="video-dialog-title">本地视频预览</h2><p id="video-dialog-description" className="helper-copy">检查视频尺寸、时长和 6 张抽样帧。文件仅在当前浏览器内读取。</p></div><button type="button" className="icon-button" aria-label="关闭视频预览" onClick={onClose}><X size={18} aria-hidden="true" /></button></div>
      <div className="dialog-content">
        <label><span className="block mb-2 text-sm text-slate-200">选择视频文件</span><input type="file" accept="video/mp4,video/quicktime,video/webm,.mp4,.m4v,.mov,.webm" className="file-input" onChange={(event) => { void selectFile(event.target.files?.[0]); event.target.value = ''; }} /></label>
        <p className="helper-copy">支持浏览器可解码的 MP4、MOV、WebM，最大 512 MB。用于重建还需要相机内参、逐帧位姿与数据划分。</p>
        {busy && <div role="status" aria-live="polite"><p className="text-sm text-cyan-200 break-all">正在读取 {filename} · {completed} / 6 帧</p><progress aria-label="视频抽样进度" max={6} value={completed} className="mt-3 w-full accent-cyan-400" /><button type="button" className="secondary-button mt-3" onClick={() => { controllerRef.current?.abort(); setBusy(false); }}>取消读取</button></div>}
        {error && <p role="alert" className="dialog-error">{error}</p>}
        {preview && !busy && <section aria-label="已读取的视频信息">
          <div className="flex flex-wrap items-start justify-between gap-3 mb-4"><div className="min-w-0"><h3 className="font-medium text-slate-100 break-all">{preview.name}</h3><p className="helper-copy mt-1 numeric">{preview.width} × {preview.height} · {preview.durationSec.toFixed(2)} s · {(preview.sizeBytes / 1024 / 1024).toFixed(1)} MB</p></div><span className="source-badge">本地视频 · 仅预览</span></div>
          <div className="video-preview-grid">{preview.frames.map((frame, index) => <figure key={index}><img src={frame.imageUrl} alt={`视频抽样帧 ${index + 1}，时间 ${frame.timestampSec.toFixed(2)} 秒`} /><figcaption><span>帧 {index + 1}</span><span>{frame.timestampSec.toFixed(2)} s</span></figcaption></figure>)}</div>
        </section>}
        {!preview && !busy && !error && <div className="flex items-center gap-3 py-6 text-slate-400"><Film size={25} aria-hidden="true" /><p className="text-sm leading-relaxed">选择一段视频后，可在这里检查抽样画面。<br />三维工作区继续显示当前合成场景。</p></div>}
      </div>
      <div className="dialog-footer"><p className="helper-copy">当前步骤只预览 RGB，不生成深度、位姿或高斯。</p><button type="button" className="secondary-button" onClick={onClose}>{busy ? '取消并关闭' : '返回工作台'}</button></div>
    </dialog>
  );
}

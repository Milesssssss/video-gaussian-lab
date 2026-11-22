export interface PreviewFrame { timestampSec: number; imageUrl: string }
export interface VideoPreview {
  name: string; sizeBytes: number; width: number; height: number; durationSec: number;
  provenance: 'local-video-preview'; frames: PreviewFrame[];
}
export const MAX_VIDEO_BYTES = 512 * 1024 * 1024;

export function validateVideoFile(file: Pick<File, 'name' | 'size' | 'type'>): void {
  if (!file.size) throw new Error('视频文件为空，请重新选择。');
  if (file.size > MAX_VIDEO_BYTES) throw new Error('预览支持最大 512 MB 的视频，请选择较短片段。');
  if (!/\.(mp4|m4v|mov|webm)$/i.test(file.name) || (file.type !== '' && !file.type.startsWith('video/'))) {
    throw new Error('请选择 MP4、MOV 或 WebM 视频。');
  }
}

export function previewTimestamps(durationSec: number, count = 6): number[] {
  if (!Number.isFinite(durationSec) || durationSec <= 0) throw new Error('无法读取视频时长，请尝试标准 MP4 文件。');
  if (!Number.isInteger(count) || count < 1 || count > 12) throw new Error('预览帧数必须在 1–12 之间。');
  return Array.from({ length: count }, (_, index) => durationSec * (index + 0.5) / count);
}

function mediaEvent(video: HTMLVideoElement, name: string, signal: AbortSignal, action: () => void): Promise<void> {
  return new Promise((resolve, reject) => {
    let timer: ReturnType<typeof setTimeout>;
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener(name, done);
      video.removeEventListener('error', failed);
      signal.removeEventListener('abort', aborted);
    };
    const done = () => { cleanup(); resolve(); };
    const failed = () => { cleanup(); reject(new Error('浏览器无法解码此视频，请尝试 H.264 MP4 或 WebM。')); };
    const aborted = () => { cleanup(); reject(new DOMException('预览已取消', 'AbortError')); };
    if (signal.aborted) { aborted(); return; }
    video.addEventListener(name, done, { once: true });
    video.addEventListener('error', failed, { once: true });
    signal.addEventListener('abort', aborted, { once: true });
    timer = setTimeout(() => { cleanup(); reject(new Error('视频解码超时，请换用较短的视频片段。')); }, 15000);
    try { action(); } catch (error) { cleanup(); reject(error); }
  });
}

/** Browser-only I/O. Never computes depth, poses, splats, or experiment metrics. */
export async function readVideoPreview(file: File, signal: AbortSignal, onProgress: (completed: number) => void): Promise<VideoPreview> {
  validateVideoFile(file);
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  const url = URL.createObjectURL(file);
  try {
    await mediaEvent(video, 'loadeddata', signal, () => { video.src = url; video.load(); });
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) throw new Error('视频没有可读取的图像轨道。');
    const durationSec = video.duration;
    const timestamps = previewTimestamps(durationSec);
    const canvas = document.createElement('canvas');
    canvas.width = Math.min(320, width);
    canvas.height = Math.max(1, Math.round(height * canvas.width / width));
    const context = canvas.getContext('2d');
    if (!context) throw new Error('当前浏览器无法创建视频预览画布。');
    const frames: PreviewFrame[] = [];
    for (const timestampSec of timestamps) {
      if (signal.aborted) throw new DOMException('预览已取消', 'AbortError');
      await mediaEvent(video, 'seeked', signal, () => { video.currentTime = timestampSec; });
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      frames.push({ timestampSec: video.currentTime, imageUrl: canvas.toDataURL('image/jpeg', 0.8) });
      onProgress(frames.length);
    }
    return { name: file.name, sizeBytes: file.size, width, height, durationSec, provenance: 'local-video-preview', frames };
  } finally {
    video.pause();
    video.removeAttribute('src');
    video.load();
    URL.revokeObjectURL(url);
  }
}

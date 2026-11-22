import type { GaussianSplat, VideoSequence } from '../types/pipeline';

/** Point positions and colors only: deliberately not a 3DGS checkpoint. */
export function exportDemoPointCloud(sequence: VideoSequence, splats: GaussianSplat[]): string {
  if (sequence.provenance !== 'synthetic') throw new Error('仅支持导出合成演示点云。');
  const rows = splats.map((splat) => {
    if (![...splat.position, ...splat.color].every(Number.isFinite)) throw new Error('点云包含非有限数值。');
    const rgb = splat.color.map((value) => Math.round(Math.min(1, Math.max(0, value)) * 255));
    return [...splat.position.map((value) => value.toFixed(6)), ...rgb].join(' ');
  });
  return [
    'ply', 'format ascii 1.0',
    'comment synthetic_demo_not_a_reconstruction_result',
    'comment point_cloud_only_no_gaussian_covariance_or_rotation',
    `element vertex ${rows.length}`, 'property float x', 'property float y', 'property float z',
    'property uchar red', 'property uchar green', 'property uchar blue', 'end_header', ...rows, '',
  ].join('\n');
}

export function downloadText(content: string, filename: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: 'application/octet-stream' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

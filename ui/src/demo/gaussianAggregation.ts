/** Synthetic scene and heuristic fusion illustration. No SHARP / DP2 inference or research metrics. */
import type {
  AggregationResult,
  CameraPose,
  ConsecutiveFeatureLink,
  FeatureAnchorNode,
  FeatureMatch2D,
  FrameData,
  FrameMetricPoint,
  GaussianSplat,
  PipelineParams,
  VideoPresetId,
  VideoSequence,
} from '../types/pipeline';
import { solveUmeyamaSim3 } from './geometry';

const FRAME_W = 160;
const FRAME_H = 96;

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function seededRandom(seed: number): () => number {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

interface CanonicalPoint {
  pos: [number, number, number];
  scale: [number, number, number];
  color: [number, number, number];
  isLandmark?: boolean;
}

function addBoxPoints(
  list: CanonicalPoint[],
  cx: number,
  cy: number,
  cz: number,
  wx: number,
  hy: number,
  dz: number,
  color: [number, number, number],
  density: number,
  rand: () => number,
) {
  for (let i = 0; i < density; i++) {
    const face = Math.floor(rand() * 6);
    let x = cx + (rand() - 0.5) * wx;
    let y = cy + (rand() - 0.5) * hy;
    let z = cz + (rand() - 0.5) * dz;
    if (face === 0) x = cx + wx * 0.5;
    else if (face === 1) x = cx - wx * 0.5;
    else if (face === 2) y = cy + hy * 0.5;
    else if (face === 3) y = cy - hy * 0.5;
    else if (face === 4) z = cz + dz * 0.5;
    else z = cz - dz * 0.5;

    const shade = 0.84 + rand() * 0.28;
    list.push({
      pos: [x, y, z],
      scale: [0.075 + rand() * 0.04, 0.075 + rand() * 0.04, 0.075 + rand() * 0.04],
      color: [
        clamp(color[0] * shade, 0, 1),
        clamp(color[1] * shade, 0, 1),
        clamp(color[2] * shade, 0, 1),
      ],
    });
  }
}

function addCylinderPoints(
  list: CanonicalPoint[],
  cx: number,
  cyBase: number,
  cz: number,
  radius: number,
  height: number,
  color: [number, number, number],
  density: number,
  rand: () => number,
) {
  for (let i = 0; i < density; i++) {
    const theta = rand() * Math.PI * 2;
    const h = rand() * height;
    const r = radius * (0.94 + rand() * 0.1);
    const shade = 0.8 + 0.35 * (0.5 + 0.5 * Math.cos(theta - 0.8));
    list.push({
      pos: [cx + Math.cos(theta) * r, cyBase + h, cz + Math.sin(theta) * r],
      scale: [0.065 + rand() * 0.03, 0.08 + rand() * 0.03, 0.065 + rand() * 0.03],
      color: [
        clamp(color[0] * shade, 0, 1),
        clamp(color[1] * shade, 0, 1),
        clamp(color[2] * shade, 0, 1),
      ],
    });
  }
}

function addSphereCluster(
  list: CanonicalPoint[],
  cx: number,
  cy: number,
  cz: number,
  radius: number,
  color: [number, number, number],
  density: number,
  rand: () => number,
) {
  for (let i = 0; i < density; i++) {
    const u = rand() * 2 - 1;
    const theta = rand() * Math.PI * 2;
    const r = radius * Math.pow(rand(), 0.25);
    const s = Math.sqrt(Math.max(0, 1 - u * u));
    const shade = 0.78 + rand() * 0.35;
    list.push({
      pos: [cx + r * s * Math.cos(theta), cy + r * u, cz + r * s * Math.sin(theta)],
      scale: [0.07 + rand() * 0.05, 0.07 + rand() * 0.05, 0.07 + rand() * 0.05],
      color: [
        clamp(color[0] * shade, 0, 1),
        clamp(color[1] * shade, 0, 1),
        clamp(color[2] * shade, 0, 1),
      ],
    });
  }
}

function buildPresetGeometry(presetId: Exclude<VideoPresetId, 'custom'>): {
  points: CanonicalPoint[];
  landmarks: [number, number, number][];
} {
  const rand = seededRandom(presetId === 'courtyard' ? 1042 : presetId === 'workbench' ? 2084 : 3099);
  const points: CanonicalPoint[] = [];
  const landmarks: [number, number, number][] = [];

  // Ground plane
  for (let gx = -3.4; gx <= 3.4; gx += 0.34) {
    for (let gz = -3.4; gz <= 3.4; gz += 0.34) {
      const checker = (Math.floor(gx + 4) + Math.floor(gz + 4)) % 2 === 0;
      const base: [number, number, number] =
        presetId === 'workbench'
          ? checker
            ? [0.22, 0.25, 0.30]
            : [0.16, 0.19, 0.23]
          : presetId === 'greenhouse'
            ? checker
              ? [0.28, 0.24, 0.20]
              : [0.23, 0.20, 0.17]
            : checker
              ? [0.38, 0.40, 0.42]
              : [0.30, 0.32, 0.35];
      points.push({
        pos: [gx + (rand() - 0.5) * 0.08, -0.02, gz + (rand() - 0.5) * 0.08],
        scale: [0.17, 0.03, 0.17],
        color: base,
      });
    }
  }

  if (presetId === 'courtyard') {
    // Architectural pavilion: stepped plinth + 4 columns + roof + central sculpture
    addBoxPoints(points, 0, 0.12, 0, 3.4, 0.24, 3.4, [0.78, 0.76, 0.72], 140, rand);
    addBoxPoints(points, 0, 0.32, 0, 2.7, 0.16, 2.7, [0.70, 0.68, 0.65], 110, rand);

    const colCoords: [number, number][] = [
      [-1.05, -1.05],
      [1.05, -1.05],
      [-1.05, 1.05],
      [1.05, 1.05],
    ];
    for (const [cx, cz] of colCoords) {
      addCylinderPoints(points, cx, 0.4, cz, 0.16, 1.75, [0.85, 0.83, 0.78], 85, rand);
      landmarks.push([cx, 0.55, cz], [cx, 1.95, cz]);
    }
    // Roof canopy
    addBoxPoints(points, 0, 2.25, 0, 2.9, 0.20, 2.9, [0.26, 0.42, 0.52], 150, rand);
    addBoxPoints(points, 0, 2.45, 0, 2.1, 0.16, 2.1, [0.20, 0.33, 0.42], 90, rand);

    // Central bronze torus/pedestal monument
    addBoxPoints(points, 0, 0.68, 0, 0.62, 0.55, 0.62, [0.30, 0.32, 0.36], 65, rand);
    addSphereCluster(points, 0, 1.32, 0, 0.34, [0.88, 0.58, 0.22], 95, rand);
    landmarks.push([0, 1.32, 0], [0, 2.25, 1.3], [0, 2.25, -1.3], [-1.3, 2.25, 0], [1.3, 2.25, 0]);

    // Flanking cypress shrubs
    addSphereCluster(points, -2.2, 0.75, -1.7, 0.48, [0.22, 0.52, 0.32], 65, rand);
    addSphereCluster(points, 2.2, 0.85, 1.6, 0.52, [0.24, 0.56, 0.34], 65, rand);
    landmarks.push([-2.2, 0.75, -1.7], [2.2, 0.85, 1.6]);
  } else if (presetId === 'workbench') {
    // Optical calibration bench with camera lens, prism, monitor frame, mug
    addBoxPoints(points, 0, 0.15, 0, 3.6, 0.30, 2.2, [0.26, 0.29, 0.34], 160, rand);
    // Telephoto lens housing
    addCylinderPoints(points, -0.75, 0.32, 0.15, 0.30, 0.95, [0.18, 0.20, 0.24], 110, rand);
    addCylinderPoints(points, -0.75, 1.25, 0.15, 0.26, 0.20, [0.12, 0.68, 0.82], 55, rand);
    // Calibration cube & target
    addBoxPoints(points, 0.35, 0.65, -0.25, 0.65, 0.65, 0.65, [0.92, 0.90, 0.84], 120, rand);
    addBoxPoints(points, 0.35, 0.65, 0.09, 0.45, 0.45, 0.04, [0.92, 0.48, 0.16], 45, rand);
    // Brass optical stand
    addCylinderPoints(points, 1.25, 0.30, 0.35, 0.10, 1.15, [0.84, 0.64, 0.24], 75, rand);
    addSphereCluster(points, 1.25, 1.55, 0.35, 0.26, [0.22, 0.72, 0.88], 70, rand);
    // Reference ceramic vessel
    addCylinderPoints(points, -1.4, 0.30, -0.45, 0.22, 0.48, [0.85, 0.36, 0.28], 65, rand);

    landmarks.push(
      [-0.75, 1.25, 0.15],
      [-0.75, 0.45, 0.15],
      [0.35, 0.98, -0.25],
      [0.35, 0.35, 0.09],
      [1.25, 1.55, 0.35],
      [1.25, 0.45, 0.35],
      [-1.4, 0.75, -0.45],
      [0.0, 0.30, 0.9],
      [-1.2, 0.30, 0.8],
      [1.2, 0.30, -0.8],
    );
  } else {
    // Botanical greenhouse walk: arched ribs, raised planters, central fountain
    for (const zOff of [-1.8, 0, 1.8]) {
      addCylinderPoints(points, -1.6, 0, zOff, 0.11, 2.2, [0.76, 0.80, 0.84], 55, rand);
      addCylinderPoints(points, 1.6, 0, zOff, 0.11, 2.2, [0.76, 0.80, 0.84], 55, rand);
      addBoxPoints(points, 0, 2.30, zOff, 3.3, 0.14, 0.14, [0.74, 0.78, 0.82], 55, rand);
      landmarks.push([-1.6, 1.8, zOff], [1.6, 1.8, zOff]);
    }
    // Raised terracotta planters + botanical foliage
    addBoxPoints(points, -1.2, 0.30, -0.5, 0.95, 0.60, 2.6, [0.68, 0.38, 0.26], 110, rand);
    addBoxPoints(points, 1.2, 0.30, 0.4, 0.95, 0.60, 2.6, [0.68, 0.38, 0.26], 110, rand);
    addSphereCluster(points, -1.2, 1.02, -1.2, 0.48, [0.22, 0.65, 0.36], 75, rand);
    addSphereCluster(points, -1.15, 0.95, 0.2, 0.42, [0.30, 0.72, 0.40], 65, rand);
    addSphereCluster(points, 1.2, 1.05, -0.3, 0.46, [0.20, 0.62, 0.34], 75, rand);
    addSphereCluster(points, 1.18, 0.92, 1.1, 0.44, [0.85, 0.44, 0.24], 70, rand);
    // Central water basin
    addCylinderPoints(points, 0, 0.0, 0.1, 0.58, 0.42, [0.16, 0.58, 0.72], 85, rand);
    landmarks.push([0, 0.42, 0.1], [-1.2, 1.02, -1.2], [1.2, 1.05, -0.3], [1.18, 0.92, 1.1]);
  }

  return { points, landmarks };
}

interface CameraBasis {
  pos: [number, number, number];
  right: [number, number, number];
  up: [number, number, number];
  forward: [number, number, number];
}

function buildCameraBasis(pos: [number, number, number], lookAt: [number, number, number]): CameraBasis {
  const fx = lookAt[0] - pos[0];
  const fy = lookAt[1] - pos[1];
  const fz = lookAt[2] - pos[2];
  const fLen = Math.hypot(fx, fy, fz) || 1;
  const forward: [number, number, number] = [fx / fLen, fy / fLen, fz / fLen];

  // world up (0, 1, 0)
  let rx = -forward[2];
  let ry = 0;
  let rz = forward[0];
  const rLen = Math.hypot(rx, ry, rz) || 1;
  rx /= rLen;
  rz /= rLen;
  const right: [number, number, number] = [rx, ry, rz];

  const ux = right[1] * forward[2] - right[2] * forward[1];
  const uy = right[2] * forward[0] - right[0] * forward[2];
  const uz = right[0] * forward[1] - right[1] * forward[0];
  const up: [number, number, number] = [ux, uy, uz];

  return { pos, right, up, forward };
}

function projectPoint(
  pt: [number, number, number],
  basis: CameraBasis,
  focalPx: number,
): { u: number; v: number; z: number } | null {
  const dx = pt[0] - basis.pos[0];
  const dy = pt[1] - basis.pos[1];
  const dz = pt[2] - basis.pos[2];

  const cx = dx * basis.right[0] + dy * basis.right[1] + dz * basis.right[2];
  const cy = dx * basis.up[0] + dy * basis.up[1] + dz * basis.up[2];
  const cz = dx * basis.forward[0] + dy * basis.forward[1] + dz * basis.forward[2];

  if (cz <= 0.35) return null;
  const u = FRAME_W * 0.5 + (cx / cz) * focalPx;
  const v = FRAME_H * 0.5 - (cy / cz) * focalPx;
  return { u, v, z: cz };
}

function renderSyntheticFrameBuffer(
  points: CanonicalPoint[],
  basis: CameraBasis,
  focalPx: number,
): {
  rgbPixels: Uint8ClampedArray;
  depthMapMeters: Float32Array;
  minDepth: number;
  maxDepth: number;
  meanDepth: number;
} {
  const rgbPixels = new Uint8ClampedArray(FRAME_W * FRAME_H * 4);
  const depthMapMeters = new Float32Array(FRAME_W * FRAME_H);

  // Sky / studio gradient background
  for (let y = 0; y < FRAME_H; y++) {
    const t = y / FRAME_H;
    const bgR = Math.round(14 + t * 14);
    const bgG = Math.round(20 + t * 18);
    const bgB = Math.round(32 + t * 22);
    const bgDepth = 12.0 - t * 4.5;
    for (let x = 0; x < FRAME_W; x++) {
      const idx = y * FRAME_W + x;
      depthMapMeters[idx] = bgDepth;
      rgbPixels[idx * 4] = bgR;
      rgbPixels[idx * 4 + 1] = bgG;
      rgbPixels[idx * 4 + 2] = bgB;
      rgbPixels[idx * 4 + 3] = 255;
    }
  }

  for (const pt of points) {
    const proj = projectPoint(pt.pos, basis, focalPx);
    if (!proj) continue;
    if (proj.u < -4 || proj.u >= FRAME_W + 4 || proj.v < -4 || proj.v >= FRAME_H + 4) continue;

    const rad = clamp(Math.round((pt.scale[0] * focalPx) / proj.z) + 1, 1, 4);
    const cu = Math.round(proj.u);
    const cv = Math.round(proj.v);

    for (let dy = -rad; dy <= rad; dy++) {
      const py = cv + dy;
      if (py < 0 || py >= FRAME_H) continue;
      for (let dx = -rad; dx <= rad; dx++) {
        const px = cu + dx;
        if (px < 0 || px >= FRAME_W) continue;
        if (dx * dx + dy * dy > rad * rad + 0.5) continue;
        const idx = py * FRAME_W + px;
        if (proj.z < depthMapMeters[idx]) {
          depthMapMeters[idx] = proj.z;
          rgbPixels[idx * 4] = Math.round(pt.color[0] * 255);
          rgbPixels[idx * 4 + 1] = Math.round(pt.color[1] * 255);
          rgbPixels[idx * 4 + 2] = Math.round(pt.color[2] * 255);
          rgbPixels[idx * 4 + 3] = 255;
        }
      }
    }
  }

  let minDepth = Infinity;
  let maxDepth = 0;
  let sumDepth = 0;
  for (let i = 0; i < depthMapMeters.length; i++) {
    const d = depthMapMeters[i];
    if (d < minDepth) minDepth = d;
    if (d > maxDepth) maxDepth = d;
    sumDepth += d;
  }

  return {
    rgbPixels,
    depthMapMeters,
    minDepth: Number.isFinite(minDepth) ? minDepth : 1.0,
    maxDepth: maxDepth || 12.0,
    meanDepth: sumDepth / depthMapMeters.length,
  };
}

export function generatePresetVideoSequence(presetId: Exclude<VideoPresetId, 'custom'>): VideoSequence {
  const { points, landmarks } = buildPresetGeometry(presetId);
  const frameCount = 16;
  const fps = 4;
  const gtFocalPx = 118;
  const frames: FrameData[] = [];
  const rand = seededRandom(presetId === 'courtyard' ? 771 : presetId === 'workbench' ? 882 : 993);

  const gtPoses: CameraPose[] = [];
  const rawPoses: CameraPose[] = [];

  for (let t = 0; t < frameCount; t++) {
    const u = t / (frameCount - 1);
    let pos: [number, number, number];
    let lookAt: [number, number, number];

    if (presetId === 'courtyard') {
      const angle = -Math.PI * 0.65 + u * Math.PI * 1.15;
      const radius = 4.35;
      pos = [Math.cos(angle) * radius, 1.85 + Math.sin(u * Math.PI) * 0.45, Math.sin(angle) * radius];
      lookAt = [0, 1.05, 0];
    } else if (presetId === 'workbench') {
      const angle = -Math.PI * 0.35 + u * Math.PI * 0.72;
      pos = [Math.sin(angle) * 3.1, 1.55 + Math.cos(u * Math.PI * 2) * 0.15, 2.75 - u * 0.45];
      lookAt = [0.05, 0.72, 0.05];
    } else {
      pos = [Math.sin(u * Math.PI * 1.4) * 0.42, 1.38 + Math.sin(u * Math.PI) * 0.12, 3.6 - u * 3.1];
      lookAt = [0, 0.85, -0.6 - u * 0.9];
    }

    gtPoses.push({
      frameIndex: t,
      timestampSec: t / fps,
      position: pos,
      lookAt,
      focalPx: gtFocalPx,
      scaleFactor: 1.0,
      driftMm: 0,
    });

    // Inject illustrative pose and scale drift into the generated scene.
    const scaleJitter = 1.0 + 0.11 * Math.sin(t * 0.9 + 0.4) + 0.012 * t;
    const driftX = (t * 0.036 * Math.cos(t * 0.38)) + (rand() - 0.5) * 0.05;
    const driftY = (t * 0.019 * Math.sin(t * 0.45)) + (rand() - 0.5) * 0.04;
    const driftZ = (t * 0.032 * Math.sin(t * 0.25)) + (rand() - 0.5) * 0.05;
    const driftMm = Math.hypot(driftX, driftY, driftZ) * 1000;

    rawPoses.push({
      frameIndex: t,
      timestampSec: t / fps,
      position: [pos[0] + driftX, pos[1] + driftY, pos[2] + driftZ],
      lookAt: [lookAt[0] + driftX * 0.45, lookAt[1] - driftY * 0.35, lookAt[2] + driftZ * 0.45],
      focalPx: gtFocalPx * (1.0 + (scaleJitter - 1.0) * 0.7),
      scaleFactor: scaleJitter,
      driftMm,
    });
  }

  let splatIdCounter = 1;
  for (let t = 0; t < frameCount; t++) {
    const gt = gtPoses[t];
    const raw = rawPoses[t];
    const basis = buildCameraBasis(gt.position, gt.lookAt);
    const rendered = renderSyntheticFrameBuffer(points, basis, gtFocalPx);

    // Sample visible splats from this frame's frustum (simulating ml-sharp single-frame Gaussian prediction)
    const rawSplats: GaussianSplat[] = [];
    for (let i = 0; i < points.length; i++) {
      const pt = points[i];
      const proj = projectPoint(pt.pos, basis, gtFocalPx);
      if (!proj) continue;
      if (proj.u < 2 || proj.u >= FRAME_W - 2 || proj.v < 2 || proj.v >= FRAME_H - 2) continue;

      // Inject deterministic per-frame jitter in the synthetic camera-ray direction.
      const rayX = pt.pos[0] - gt.position[0];
      const rayY = pt.pos[1] - gt.position[1];
      const rayZ = pt.pos[2] - gt.position[2];

      const rawX = raw.position[0] + rayX * raw.scaleFactor + (rand() - 0.5) * 0.028;
      const rawY = raw.position[1] + rayY * raw.scaleFactor + (rand() - 0.5) * 0.028;
      const rawZ = raw.position[2] + rayZ * raw.scaleFactor + (rand() - 0.5) * 0.028;

      const residual = Math.hypot(rawX - pt.pos[0], rawY - pt.pos[1], rawZ - pt.pos[2]);

      rawSplats.push({
        id: splatIdCounter++,
        position: [rawX, rawY, rawZ],
        scale: [pt.scale[0] * raw.scaleFactor, pt.scale[1] * raw.scaleFactor, pt.scale[2] * raw.scaleFactor],
        color: pt.color,
        opacity: 0.78,
        frameIndex: t,
        depthMeters: proj.z,
        residualError: residual,
        mergedCount: 1,
      });
    }

    // Build 2D-3D feature matches to next frame
    const matchesToNext: FeatureMatch2D[] = [];
    if (t < frameCount - 1) {
      const nextBasis = buildCameraBasis(gtPoses[t + 1].position, gtPoses[t + 1].lookAt);
      landmarks.forEach((lm, idx) => {
        const p1 = projectPoint(lm, basis, gtFocalPx);
        const p2 = projectPoint(lm, nextBasis, gtFocalPx);
        if (p1 && p2 && p1.u >= 4 && p1.u <= FRAME_W - 4 && p1.v >= 4 && p1.v <= FRAME_H - 4 && p2.u >= 4 && p2.u <= FRAME_W - 4 && p2.v >= 4 && p2.v <= FRAME_H - 4) {
          matchesToNext.push({
            id: idx,
            u1: p1.u,
            v1: p1.v,
            u2: p2.u,
            v2: p2.v,
            landmark3D: lm,
            reprojErrorPx: 2.1 + rand() * 3.4,
            inlier: idx % 9 !== 8,
          });
        }
      });
    }

    frames.push({
      frameIndex: t,
      timestampSec: t / fps,
      width: FRAME_W,
      height: FRAME_H,
      rgbPixels: rendered.rgbPixels,
      depthMapMeters: rendered.depthMapMeters,
      minDepth: rendered.minDepth,
      maxDepth: rendered.maxDepth,
      meanDepth: rendered.meanDepth,
      estimatedFocalPx: gtFocalPx * (1.0 + (rand() - 0.5) * 0.012),
      rawPose: raw,
      gtPose: gt,
      rawSplats,
      matchesToNext,
    });
  }

  const meta =
    presetId === 'courtyard'
      ? {
          title: '建筑庭院环绕 (Courtyard Orbit)',
          subtitle: '360° 柱廊与中央雕塑环绕轨迹 · 强多视角回环特征',
        }
      : presetId === 'workbench'
        ? {
            title: '实验桌面推轨 (Optical Bench Dolly)',
            subtitle: '近景相机镜头与标定体块 · 高频边缘与深度跃变测试',
          }
        : {
            title: '植物温室穿行 (Botanical Atrium Walk)',
            subtitle: '前向平移穿行序列 · 遮挡层次与连续光流视差',
          };

  return {
    id: presetId,
    provenance: 'synthetic',
    title: meta.title,
    subtitle: meta.subtitle,
    fps,
    durationSec: (frameCount - 1) / fps,
    resolutionLabel: '160×96 · 合成关键帧',
    frames,
  };
}

/**
 * Point alignment for synthetic fixtures, with a bounded scale.
 * This helper is not a cross-view optimizer and does not invoke a model.
 */
export function solveSim3CentroidScale(
  sourcePts: [number, number, number][],
  targetPts: [number, number, number][],
): { scale: number; rotation: number[][]; translation: [number, number, number]; rmse: number } {
  return solveUmeyamaSim3(sourcePts, targetPts);
}

/**
 * Synthetic fusion illustration using a spatial hash, Euclidean distance and RGB proximity.
 * The scalar distance normalization is not a full covariance / Mahalanobis test.
 * This heuristic is not the Phase 1 visibility or admission implementation.
 */
export function fuseOverlappingGaussians(
  splats: GaussianSplat[],
  mergeRadiusMeters: number,
): GaussianSplat[] {
  if (splats.length === 0 || mergeRadiusMeters <= 0.005) return splats;

  const cellSize = Math.max(0.04, mergeRadiusMeters);
  const grid = new Map<string, GaussianSplat[]>();
  const fused: GaussianSplat[] = [];

  const keyFor = (x: number, y: number, z: number) =>
    `${Math.floor(x / cellSize)},${Math.floor(y / cellSize)},${Math.floor(z / cellSize)}`;

  for (const s of splats) {
    const gx = Math.floor(s.position[0] / cellSize);
    const gy = Math.floor(s.position[1] / cellSize);
    const gz = Math.floor(s.position[2] / cellSize);

    let mergedTarget: GaussianSplat | null = null;
    for (let dx = -1; dx <= 1 && !mergedTarget; dx++) {
      for (let dy = -1; dy <= 1 && !mergedTarget; dy++) {
        for (let dz = -1; dz <= 1 && !mergedTarget; dz++) {
          const bucket = grid.get(`${gx + dx},${gy + dy},${gz + dz}`);
          if (!bucket) continue;
          for (const cand of bucket) {
            const distSq =
              (cand.position[0] - s.position[0]) ** 2 +
              (cand.position[1] - s.position[1]) ** 2 +
              (cand.position[2] - s.position[2]) ** 2;
            const sigma = Math.max(0.03, (cand.scale[0] + s.scale[0]) * 0.5);
            const mahalanobisNorm = Math.sqrt(distSq) / Math.max(mergeRadiusMeters, sigma);
            if (mahalanobisNorm <= 1.0) {
              const colorDiff =
                Math.abs(cand.color[0] - s.color[0]) +
                Math.abs(cand.color[1] - s.color[1]) +
                Math.abs(cand.color[2] - s.color[2]);
              if (colorDiff < 0.42) {
                mergedTarget = cand;
                break;
              }
            }
          }
        }
      }
    }

    if (mergedTarget) {
      const w1 = mergedTarget.mergedCount;
      const w2 = 1;
      const total = w1 + w2;
      mergedTarget.position = [
        (mergedTarget.position[0] * w1 + s.position[0]) / total,
        (mergedTarget.position[1] * w1 + s.position[1]) / total,
        (mergedTarget.position[2] * w1 + s.position[2]) / total,
      ];
      mergedTarget.color = [
        (mergedTarget.color[0] * w1 + s.color[0]) / total,
        (mergedTarget.color[1] * w1 + s.color[1]) / total,
        (mergedTarget.color[2] * w1 + s.color[2]) / total,
      ];
      mergedTarget.scale = [
        (mergedTarget.scale[0] * w1 + s.scale[0]) / total,
        (mergedTarget.scale[1] * w1 + s.scale[1]) / total,
        (mergedTarget.scale[2] * w1 + s.scale[2]) / total,
      ];
      mergedTarget.opacity = clamp(1 - (1 - mergedTarget.opacity) * (1 - s.opacity * 0.45), 0.3, 0.96);
      mergedTarget.residualError = Math.min(mergedTarget.residualError, s.residualError);
      mergedTarget.mergedCount = total;
    } else {
      const clone: GaussianSplat = {
        ...s,
        position: [...s.position],
        scale: [...s.scale],
        color: [...s.color],
      };
      fused.push(clone);
      const k = keyFor(s.position[0], s.position[1], s.position[2]);
      const bucket = grid.get(k);
      if (bucket) bucket.push(clone);
      else grid.set(k, [clone]);
    }
  }

  return fused;
}

export function runAggregationPipeline(
  sequence: VideoSequence,
  params: PipelineParams,
): AggregationResult {
  // GT-guided interpolation is deliberately confined to generated demo data.
  // Never use this function for imported captures, model predictions or experiment reports.
  if (sequence.provenance !== 'synthetic') {
    throw new Error('Demo aggregation accepts only synthetic sequences. Real reconstruction results require a separate viewer adapter.');
  }
  const maxFrameIdx = clamp(params.maxActiveFrame, 0, sequence.frames.length - 1);
  const activeFrames = sequence.frames.slice(0, maxFrameIdx + 1);

  const depthWeight = params.mode === 'raw_sharp' ? 0 : params.depthTemporalWeight;
  const globalWeight = params.mode === 'global_fused' ? params.globalConstraintWeight : 0;

  const adjustedSplats: GaussianSplat[] = [];
  const adjustedPoses: CameraPose[] = [];
  const featureTracks: AggregationResult['featureTracks'] = [];
  const perFrameSeries: FrameMetricPoint[] = [];

  let rawSplatTotal = 0;

  for (let i = 0; i < activeFrames.length; i++) {
    const frame = activeFrames[i];
    const gt = frame.gtPose;
    const raw = frame.rawPose;
    rawSplatTotal += frame.rawSplats.length;

    // 1. Illustrate suppression of generated scale noise with a hand-written slider formula.
    const effectiveScale = 1.0 + (raw.scaleFactor - 1.0) * (1.0 - depthWeight * 0.88);

    // 2. Interpolate toward the known synthetic pose. No bundle adjustment or pose estimation runs here.
    const poseRetention = (1.0 - depthWeight * 0.35) * (1.0 - globalWeight * 0.92);
    const adjPos: [number, number, number] = [
      gt.position[0] + (raw.position[0] - gt.position[0]) * poseRetention,
      gt.position[1] + (raw.position[1] - gt.position[1]) * poseRetention,
      gt.position[2] + (raw.position[2] - gt.position[2]) * poseRetention,
    ];
    const adjLook: [number, number, number] = [
      gt.lookAt[0] + (raw.lookAt[0] - gt.lookAt[0]) * poseRetention,
      gt.lookAt[1] + (raw.lookAt[1] - gt.lookAt[1]) * poseRetention,
      gt.lookAt[2] + (raw.lookAt[2] - gt.lookAt[2]) * poseRetention,
    ];

    const adjDriftMm = Math.hypot(
      adjPos[0] - gt.position[0],
      adjPos[1] - gt.position[1],
      adjPos[2] - gt.position[2],
    ) * 1000;

    adjustedPoses.push({
      frameIndex: frame.frameIndex,
      timestampSec: frame.timestampSec,
      position: adjPos,
      lookAt: adjLook,
      focalPx: gt.focalPx + (raw.focalPx - gt.focalPx) * (1.0 - depthWeight * 0.85),
      scaleFactor: effectiveScale,
      driftMm: adjDriftMm,
    });

    // Transform generated demo splats according to the synthetic interpolation above.
    for (const sp of frame.rawSplats) {
      // Recover gt-relative ray from rawPose and re-project under corrected pose & scale
      const relX = (sp.position[0] - raw.position[0]) / raw.scaleFactor;
      const relY = (sp.position[1] - raw.position[1]) / raw.scaleFactor;
      const relZ = (sp.position[2] - raw.position[2]) / raw.scaleFactor;

      const worldX = adjPos[0] + relX * effectiveScale;
      const worldY = adjPos[1] + relY * effectiveScale;
      const worldZ = adjPos[2] + relZ * effectiveScale;

      const gtWorldX = gt.position[0] + relX;
      const gtWorldY = gt.position[1] + relY;
      const gtWorldZ = gt.position[2] + relZ;

      const err = Math.hypot(worldX - gtWorldX, worldY - gtWorldY, worldZ - gtWorldZ);

      adjustedSplats.push({
        ...sp,
        position: [worldX, worldY, worldZ],
        scale: [
          (sp.scale[0] / raw.scaleFactor) * effectiveScale,
          (sp.scale[1] / raw.scaleFactor) * effectiveScale,
          (sp.scale[2] / raw.scaleFactor) * effectiveScale,
        ],
        residualError: err,
      });
    }

    // Feature track vectors for 3D visualization
    if (i < activeFrames.length - 1) {
      for (const m of frame.matchesToNext) {
        const offsetErr = (1.0 - globalWeight * 0.90) * (1.0 - depthWeight * 0.30) * 0.14;
        featureTracks.push({
          from: [adjPos[0], adjPos[1], adjPos[2]],
          to: [
            m.landmark3D[0] + (m.inlier ? offsetErr : offsetErr * 1.8),
            m.landmark3D[1] + (m.inlier ? offsetErr * 0.5 : -offsetErr),
            m.landmark3D[2] - (m.inlier ? offsetErr * 0.4 : offsetErr * 1.5),
          ],
          errorMeters: m.inlier ? offsetErr : offsetErr * 2.1,
          inlier: m.inlier,
        });
      }
    }

    // Synthetic response curves for the demo controls; not measured ablation results.
    const rawAteMm = raw.driftMm;
    const depthProAteMm = rawAteMm * (1.0 - 0.36 * params.depthTemporalWeight);
    const fusedAteMm = depthProAteMm * (1.0 - 0.88 * params.globalConstraintWeight);

    const rawJitterCm = Math.abs(raw.scaleFactor - 1.0) * 145;
    const depthProJitterCm = rawJitterCm * (1.0 - 0.86 * params.depthTemporalWeight);
    const fusedJitterCm = depthProJitterCm * (1.0 - 0.52 * params.globalConstraintWeight);

    const rawReprojPx = 4.2 + (rawAteMm / 18);
    const fusedReprojPx = Math.max(0.45, rawReprojPx * (1.0 - 0.35 * params.depthTemporalWeight) * (1.0 - 0.78 * params.globalConstraintWeight));

    perFrameSeries.push({
      frameIndex: frame.frameIndex,
      rawAteMm,
      depthProAteMm,
      fusedAteMm,
      rawJitterCm,
      depthProJitterCm,
      fusedJitterCm,
      rawReprojPx,
      fusedReprojPx,
      rawSplatCount: rawSplatTotal,
      fusedSplatCount: Math.round(rawSplatTotal * (1 - 0.48 * Math.min(1, params.mergeRadiusMeters / 0.22))),
    });
  }

  const outputSplats =
    params.mode === 'global_fused'
      ? fuseOverlappingGaussians(adjustedSplats, params.mergeRadiusMeters)
      : adjustedSplats;

  const meanAteMm =
    adjustedPoses.reduce((acc, p) => acc + p.driftMm, 0) / Math.max(1, adjustedPoses.length);

  const depthJitterCm =
    adjustedPoses.reduce((acc, p) => acc + Math.abs(p.scaleFactor - 1.0) * 145, 0) /
    Math.max(1, adjustedPoses.length);

  const meanReprojPx =
    perFrameSeries.reduce(
      (acc, pt) => acc + (params.mode === 'global_fused' ? pt.fusedReprojPx : params.mode === 'depth_pro_2l' ? pt.rawReprojPx * 0.65 : pt.rawReprojPx),
      0,
    ) / Math.max(1, perFrameSeries.length);
  // Build consecutive-frame matched global feature links & anchor summaries
  const consecutiveFeatureLinks: ConsecutiveFeatureLink[] = [];
  const anchorAccumulator = new Map<
    number,
    {
      posSum: [number, number, number];
      gapSum: number;
      inlierCount: number;
      count: number;
    }
  >();

  for (let i = 0; i < activeFrames.length - 1; i++) {
    const frameA = activeFrames[i];
    const frameB = activeFrames[i + 1];
    const adjA = adjustedPoses[i];
    const adjB = adjustedPoses[i + 1];

    for (const m of frameA.matchesToNext) {
      const lm = m.landmark3D;
      // Deterministic micro-jitter per (frame, featureId) for illustrative per-frame observation noise
      const jA = ((m.id * 17 + frameA.frameIndex * 31) % 11 - 5) * 0.0045;
      const jB = ((m.id * 17 + frameB.frameIndex * 31) % 11 - 5) * 0.0045;
      const noiseScale = 1.0 - globalWeight * 0.72;

      // Relative ray from GT camera to 3D landmark
      const relAx = lm[0] - frameA.gtPose.position[0];
      const relAy = lm[1] - frameA.gtPose.position[1];
      const relAz = lm[2] - frameA.gtPose.position[2];

      const relBx = lm[0] - frameB.gtPose.position[0];
      const relBy = lm[1] - frameB.gtPose.position[1];
      const relBz = lm[2] - frameB.gtPose.position[2];

      const outlierBoost = m.inlier ? 1.0 : 1.75;

      // Unaligned raw synthetic 3D positions in consecutive frames t and t+1
      const rawFrom: [number, number, number] = [
        frameA.rawPose.position[0] + relAx * frameA.rawPose.scaleFactor + jA * outlierBoost,
        frameA.rawPose.position[1] + relAy * frameA.rawPose.scaleFactor - jA * 0.6 * outlierBoost,
        frameA.rawPose.position[2] + relAz * frameA.rawPose.scaleFactor + jA * 0.8 * outlierBoost,
      ];
      const rawTo: [number, number, number] = [
        frameB.rawPose.position[0] + relBx * frameB.rawPose.scaleFactor + jB * outlierBoost,
        frameB.rawPose.position[1] + relBy * frameB.rawPose.scaleFactor - jB * 0.6 * outlierBoost,
        frameB.rawPose.position[2] + relBz * frameB.rawPose.scaleFactor + jB * 0.8 * outlierBoost,
      ];

      // Current pipeline aligned 3D positions in consecutive frames t and t+1
      const from: [number, number, number] = [
        adjA.position[0] + relAx * adjA.scaleFactor + jA * noiseScale * outlierBoost,
        adjA.position[1] + relAy * adjA.scaleFactor - jA * 0.6 * noiseScale * outlierBoost,
        adjA.position[2] + relAz * adjA.scaleFactor + jA * 0.8 * noiseScale * outlierBoost,
      ];
      const to: [number, number, number] = [
        adjB.position[0] + relBx * adjB.scaleFactor + jB * noiseScale * outlierBoost,
        adjB.position[1] + relBy * adjB.scaleFactor - jB * 0.6 * noiseScale * outlierBoost,
        adjB.position[2] + relBz * adjB.scaleFactor + jB * 0.8 * noiseScale * outlierBoost,
      ];

      const gapMeters = Math.hypot(to[0] - from[0], to[1] - from[1], to[2] - from[2]);
      const pullA = Math.hypot(from[0] - rawFrom[0], from[1] - rawFrom[1], from[2] - rawFrom[2]);
      const pullB = Math.hypot(to[0] - rawTo[0], to[1] - rawTo[1], to[2] - rawTo[2]);
      const pullMeters = (pullA + pullB) * 0.5;

      const anchorPos: [number, number, number] = [
        (from[0] + to[0]) * 0.5,
        (from[1] + to[1]) * 0.5,
        (from[2] + to[2]) * 0.5,
      ];

      consecutiveFeatureLinks.push({
        featureId: m.id,
        frameFrom: frameA.frameIndex,
        frameTo: frameB.frameIndex,
        from,
        to,
        rawFrom,
        rawTo,
        anchorPos,
        gapMeters,
        pullMeters,
        inlier: m.inlier,
      });

      const acc = anchorAccumulator.get(m.id);
      if (acc) {
        acc.posSum[0] += anchorPos[0];
        acc.posSum[1] += anchorPos[1];
        acc.posSum[2] += anchorPos[2];
        acc.gapSum += gapMeters;
        acc.inlierCount += m.inlier ? 1 : 0;
        acc.count += 1;
      } else {
        anchorAccumulator.set(m.id, {
          posSum: [anchorPos[0], anchorPos[1], anchorPos[2]],
          gapSum: gapMeters,
          inlierCount: m.inlier ? 1 : 0,
          count: 1,
        });
      }
    }
  }

  const featureAnchors: FeatureAnchorNode[] = [];
  anchorAccumulator.forEach((acc, featureId) => {
    featureAnchors.push({
      featureId,
      position: [
        acc.posSum[0] / acc.count,
        acc.posSum[1] / acc.count,
        acc.posSum[2] / acc.count,
      ],
      observationCount: acc.count + 1,
      meanGapMeters: acc.gapSum / acc.count,
      inlierRatio: acc.inlierCount / acc.count,
    });
  });

  const inlierLinks = consecutiveFeatureLinks.filter((l) => l.inlier);
  const meanFeatureGapMm =
    inlierLinks.length > 0
      ? (inlierLinks.reduce((acc, l) => acc + l.gapMeters, 0) / inlierLinks.length) * 1000
      : 0;

  return {
    splats: outputSplats,
    poses: adjustedPoses,
    featureTracks,
    consecutiveFeatureLinks,
    featureAnchors,
    metrics: {
      activeFrames: activeFrames.length,
      rawSplatCount: rawSplatTotal,
      outputSplatCount: outputSplats.length,
      compressionRatio: rawSplatTotal > 0 ? outputSplats.length / rawSplatTotal : 1,
      meanAteMm,
      meanReprojPx,
      depthJitterCm,
      meanFeatureGapMm,
      inlierLinkCount: inlierLinks.length,
    },
    perFrameSeries,
  };
}

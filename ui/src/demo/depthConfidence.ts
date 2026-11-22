/** Heuristic heatmap for synthetic UI data; not model confidence or a calibrated probability. */
import type { FrameData } from '../types/pipeline';
import { computeDepthEdgeGradientJsfeat } from './geometry';

export interface DepthConfidenceResult {
  frameIndex: number;
  width: number;
  height: number;
  /** Per-pixel heuristic demo score in [0, 1]; not a probability */
  confidenceMap: Float32Array;
  /** Color-coded RGBA heatmap buffer (w * h * 4) ready for DataTexture / Canvas */
  rgba: Uint8Array;
  /** Mean heuristic score across all pixels in [0, 1] */
  meanConfidence: number;
  /** Fraction of pixels with a demo score >= 0.72 */
  highConfidenceRatio: number;
  /** Fraction of pixels with a demo score < 0.42; not a measured error probability */
  lowConfidenceRatio: number;
  /** Mean Sobel depth gradient |grad D| in meters/px */
  meanEdgeGradient: number;
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

/**
 * Maps a synthetic demo score c in [0, 1] to an RGB triple in [0, 1]:
 *   0.00 - 0.35 : Rose / Crimson (low illustrative stability)
 *   0.35 - 0.68 : Amber / Gold   (intermediate score)
 *   0.68 - 1.00 : Cyan -> Emerald (high illustrative stability)
 */
export function confidenceToRgb01(confidence: number): [number, number, number] {
  const c = clamp(confidence, 0, 1);
  if (c < 0.35) {
    const k = c / 0.35;
    return [
      0.95 + k * 0.01,
      0.24 + k * 0.38,
      0.36 - k * 0.32,
    ];
  }
  if (c < 0.68) {
    const k = (c - 0.35) / 0.33;
    return [
      0.96 - k * 0.90,
      0.62 + k * 0.14,
      0.04 + k * 0.80,
    ];
  }
  const k = (c - 0.68) / 0.32;
  return [
    0.06 + k * 0.04,
    0.76 + k * 0.12,
    0.84 - k * 0.32,
  ];
}

/**
 * Computes an illustrative heatmap using a Sobel edge heuristic, scene-range prior,
 * unwarped adjacent-frame depth differences, and proximity to synthetic features.
 * The slider adds a hand-written stability bonus. No model inference, calibrated
 * uncertainty, visibility test, or verification of real depth accuracy is performed.
 */
export function computeDemoDepthConfidenceMap(
  frame: FrameData,
  prevFrame: FrameData | null,
  depthTemporalWeight: number,
): DepthConfidenceResult {
  const w = frame.width;
  const h = frame.height;
  const n = w * h;
  const depth = frame.depthMapMeters;

  const edgeGrad = computeDepthEdgeGradientJsfeat(depth, w, h);
  const confidenceMap = new Float32Array(n);
  const rgba = new Uint8Array(n * 4);

  const hasMatchingPrev =
    prevFrame !== null &&
    prevFrame.width === w &&
    prevFrame.height === h &&
    prevFrame.depthMapMeters.length === n;

  // The demo slider reduces penalties by construction; this is not a measured model effect.
  const lockFactor = clamp(depthTemporalWeight, 0, 1);
  const scaleDrift = Math.abs(frame.rawPose.scaleFactor - 1.0);
  const effectiveDriftPenalty = scaleDrift * (1.0 - lockFactor * 0.82);
  const monocularUncertainty = (1.0 - lockFactor) * 0.16;

  // Synthetic feature proximity supplies an illustrative local bonus.
  const anchors = frame.matchesToNext.filter((m) => m.inlier);

  let sumConf = 0;
  let sumGrad = 0;
  let highCount = 0;
  let lowCount = 0;

  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const idx = row + x;
      const d = depth[idx];
      const grad = edgeGrad[idx];
      sumGrad += grad;

      // 1. Sharp depth discontinuities (|grad D|) indicate occlusion boundaries / flying splat halos
      const edgeReliability = Math.exp(-grad * 2.15);

      // 2. Metric range reliability: peak in [1.0m, 6.2m], smooth rolloff into far-field/sky background
      let rangeReliability = 1.0;
      if (d < 0.85) {
        rangeReliability = clamp(d / 0.85, 0.25, 1.0);
      } else if (d > 6.2) {
        rangeReliability = clamp(1.0 - (d - 6.2) / 6.8, 0.22, 1.0);
      }

      // 3. Same-pixel depth difference without motion compensation (demo only).
      let temporalReliability = 1.0 - clamp(effectiveDriftPenalty * 1.4, 0, 0.35);
      if (hasMatchingPrev && prevFrame) {
        const prevD = prevFrame.depthMapMeters[idx];
        const relDelta = Math.abs(d - prevD) / Math.max(1.0, Math.min(d, prevD));
        const dampedDelta = relDelta * (1.0 - lockFactor * 0.6);
        temporalReliability *= clamp(Math.exp(-dampedDelta * 1.5), 0.32, 1.0);
      }

      // 4. Hand-written feature-proximity bonus; it does not verify depth accuracy.
      let anchorBoost = 0;
      for (let a = 0; a < anchors.length; a++) {
        const dx = x - anchors[a].u1;
        const dy = y - anchors[a].v1;
        const distSq = dx * dx + dy * dy;
        if (distSq < 120) {
          const boost = 0.14 * Math.exp(-distSq / 42);
          if (boost > anchorBoost) anchorBoost = boost;
        }
      }

      const conf = clamp(
        edgeReliability * 0.46 +
          rangeReliability * 0.28 +
          temporalReliability * 0.26 +
          anchorBoost +
          lockFactor * 0.08 -
          monocularUncertainty -
          effectiveDriftPenalty * 0.45,
        0.04,
        0.99,
      );

      confidenceMap[idx] = conf;
      sumConf += conf;
      if (conf >= 0.72) highCount++;
      if (conf < 0.42) lowCount++;

      const [r, g, b] = confidenceToRgb01(conf);
      const o = idx * 4;
      rgba[o] = Math.round(r * 255);
      rgba[o + 1] = Math.round(g * 255);
      rgba[o + 2] = Math.round(b * 255);
      // Higher alpha on both high-confidence foreground & high-risk edge halos, softer in far sky
      rgba[o + 3] = Math.round(clamp(0.55 + (1 - rangeReliability) * 0.1 + (1 - edgeReliability) * 0.3, 0.55, 0.92) * 255);
    }
  }

  return {
    frameIndex: frame.frameIndex,
    width: w,
    height: h,
    confidenceMap,
    rgba,
    meanConfidence: sumConf / Math.max(1, n),
    highConfidenceRatio: highCount / Math.max(1, n),
    lowConfidenceRatio: lowCount / Math.max(1, n),
    meanEdgeGradient: sumGrad / Math.max(1, n),
  };
}

/**
 * Colors a synthetic splat using depth, generated residual and a slider bonus.
 * This is a heuristic illustration, not a model-produced confidence value.
 */
export function computeSplatDepthConfidence(
  depthMeters: number,
  residualError: number,
  depthTemporalWeight: number,
): number {
  const rangeFactor =
    depthMeters < 0.85
      ? clamp(depthMeters / 0.85, 0.25, 1.0)
      : depthMeters > 6.5
        ? clamp(1.0 - (depthMeters - 6.5) / 6.5, 0.2, 1.0)
        : 1.0;
  const residualFactor = Math.exp(-residualError * 6.5);
  const lockBonus = clamp(depthTemporalWeight, 0, 1) * 0.14;
  return clamp(rangeFactor * 0.45 + residualFactor * 0.45 + lockBonus, 0.05, 0.99);
}

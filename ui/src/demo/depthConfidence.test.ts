import test from 'node:test';
import assert from 'node:assert/strict';
import {
  computeDemoDepthConfidenceMap,
  computeSplatDepthConfidence,
  confidenceToRgb01,
} from './depthConfidence';
import { generatePresetVideoSequence } from './gaussianAggregation';
import type { FrameData } from '../types/pipeline';

test('confidenceToRgb01 maps low confidence to rose/red and high confidence to emerald/cyan', () => {
  const low = confidenceToRgb01(0.05);
  const mid = confidenceToRgb01(0.45);
  const high = confidenceToRgb01(0.95);

  // Low confidence is strongly red-dominant
  assert.ok(low[0] > 0.85 && low[0] > low[1] * 2);
  // High confidence is green/cyan dominant with low red
  assert.ok(high[1] > 0.8 && high[0] < 0.2);
  // Mid confidence stays within [0, 1]
  assert.ok(mid.every((v) => v >= 0 && v <= 1));
});

test('computeDemoDepthConfidenceMap lowers confidence at sharp depth discontinuities vs smooth surfaces', () => {
  const w = 40;
  const h = 30;
  const smoothDepth = new Float32Array(w * h).fill(3.2);
  const steppedDepth = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      // Sharp 4m depth cliff down the middle
      steppedDepth[y * w + x] = x < w / 2 ? 2.0 : 6.0;
    }
  }

  const baseFrame = (depthMapMeters: Float32Array): FrameData => ({
    frameIndex: 1,
    timestampSec: 0.25,
    width: w,
    height: h,
    rgbPixels: new Uint8ClampedArray(w * h * 4),
    depthMapMeters,
    minDepth: 2.0,
    maxDepth: 6.0,
    meanDepth: 3.5,
    estimatedFocalPx: 118,
    rawPose: {
      frameIndex: 1,
      timestampSec: 0.25,
      position: [0, 1, 3],
      lookAt: [0, 1, 0],
      focalPx: 118,
      scaleFactor: 1.05,
      driftMm: 20,
    },
    gtPose: {
      frameIndex: 1,
      timestampSec: 0.25,
      position: [0, 1, 3],
      lookAt: [0, 1, 0],
      focalPx: 118,
      scaleFactor: 1.0,
      driftMm: 0,
    },
    rawSplats: [],
    matchesToNext: [],
  });

  const smoothRes = computeDemoDepthConfidenceMap(baseFrame(smoothDepth), null, 0.9);
  const cliffRes = computeDemoDepthConfidenceMap(baseFrame(steppedDepth), null, 0.9);

  assert.equal(smoothRes.rgba.length, w * h * 4);
  const centerIdx = Math.floor(h / 2) * w + Math.floor(w / 2);
  assert.ok(
    cliffRes.confidenceMap[centerIdx] < smoothRes.confidenceMap[centerIdx] - 0.25,
    `Expected edge confidence (${cliffRes.confidenceMap[centerIdx]}) to be much lower than smooth (${smoothRes.confidenceMap[centerIdx]})`,
  );
  assert.ok(cliffRes.meanEdgeGradient > smoothRes.meanEdgeGradient);
});

test('computeDemoDepthConfidenceMap applies the synthetic slider bonus without claiming measured confidence', () => {
  const seq = generatePresetVideoSequence('courtyard');
  const unlocked = computeDemoDepthConfidenceMap(seq.frames[6], seq.frames[5], 0.0);
  const locked = computeDemoDepthConfidenceMap(seq.frames[6], seq.frames[5], 0.95);

  assert.ok(locked.meanConfidence > unlocked.meanConfidence);
  assert.ok(locked.highConfidenceRatio >= unlocked.highConfidenceRatio);
});

test('computeSplatDepthConfidence decreases monotonically with residual error', () => {
  const cGood = computeSplatDepthConfidence(3.0, 0.01, 0.9);
  const cBad = computeSplatDepthConfidence(3.0, 0.25, 0.9);
  assert.ok(cGood > cBad + 0.25);
});

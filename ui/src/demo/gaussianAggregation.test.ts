import test from 'node:test';
import assert from 'node:assert/strict';
import {
  fuseOverlappingGaussians,
  generatePresetVideoSequence,
  runAggregationPipeline,
  solveSim3CentroidScale,
} from './gaussianAggregation';
import type { GaussianSplat, VideoSequence } from '../types/pipeline';

test('generatePresetVideoSequence produces marked synthetic frames with scene depth and matches', () => {
  const seq = generatePresetVideoSequence('courtyard');
  assert.equal(seq.provenance, 'synthetic');
  assert.equal(seq.frames.length, 16);
  assert.ok(seq.frames[0].rawSplats.length > 200);
  assert.ok(seq.frames[0].matchesToNext.length > 3);
  assert.ok(seq.frames[0].minDepth > 0 && seq.frames[0].maxDepth >= seq.frames[0].minDepth);
});

test('solveSim3CentroidScale recovers exact isotropic scale and translation', () => {
  const src: [number, number, number][] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 3],
  ];
  const target: [number, number, number][] = src.map(([x, y, z]) => [
    x * 1.25 + 2.0,
    y * 1.25 - 1.5,
    z * 1.25 + 0.5,
  ]);
  const res = solveSim3CentroidScale(src, target);
  assert.ok(Math.abs(res.scale - 1.25) < 1e-6);
  assert.ok(Math.abs(res.translation[0] - 2.0) < 1e-6);
  assert.ok(Math.abs(res.translation[1] + 1.5) < 1e-6);
  assert.ok(res.rmse < 1e-6);
});

test('fuseOverlappingGaussians merges close identically colored splats and boosts opacity', () => {
  const s1: GaussianSplat = {
    id: 1,
    position: [1.0, 2.0, 3.0],
    scale: [0.1, 0.1, 0.1],
    color: [0.5, 0.6, 0.7],
    opacity: 0.7,
    frameIndex: 0,
    depthMeters: 3.0,
    residualError: 0.08,
    mergedCount: 1,
  };
  const s2: GaussianSplat = {
    id: 2,
    position: [1.02, 2.01, 2.99],
    scale: [0.1, 0.1, 0.1],
    color: [0.52, 0.58, 0.69],
    opacity: 0.7,
    frameIndex: 1,
    depthMeters: 3.0,
    residualError: 0.04,
    mergedCount: 1,
  };
  const s3: GaussianSplat = {
    id: 3,
    position: [4.5, 2.0, 3.0],
    scale: [0.1, 0.1, 0.1],
    color: [0.9, 0.1, 0.1],
    opacity: 0.7,
    frameIndex: 1,
    depthMeters: 4.5,
    residualError: 0.05,
    mergedCount: 1,
  };

  const fused = fuseOverlappingGaussians([s1, s2, s3], 0.15);
  assert.equal(fused.length, 2);
  assert.equal(fused[0].mergedCount, 2);
  assert.ok(fused[0].opacity > 0.7);
});

test('synthetic interpolation responds to demo sliders; this is not model-performance validation', () => {
  const seq = generatePresetVideoSequence('courtyard');

  const rawRes = runAggregationPipeline(seq, {
    mode: 'raw_sharp',
    depthTemporalWeight: 0,
    globalConstraintWeight: 0,
    mergeRadiusMeters: 0,
    maxActiveFrame: 15,
  });

  const depthRes = runAggregationPipeline(seq, {
    mode: 'depth_pro_2l',
    depthTemporalWeight: 0.9,
    globalConstraintWeight: 0,
    mergeRadiusMeters: 0,
    maxActiveFrame: 15,
  });

  const fusedRes = runAggregationPipeline(seq, {
    mode: 'global_fused',
    depthTemporalWeight: 0.9,
    globalConstraintWeight: 0.92,
    mergeRadiusMeters: 0.16,
    maxActiveFrame: 15,
  });

  assert.ok(depthRes.metrics.depthJitterCm < rawRes.metrics.depthJitterCm);
  assert.ok(depthRes.metrics.meanAteMm < rawRes.metrics.meanAteMm);
  assert.ok(fusedRes.metrics.meanAteMm < depthRes.metrics.meanAteMm);
  assert.ok(fusedRes.metrics.outputSplatCount < rawRes.metrics.outputSplatCount);
});

test('consecutiveFeatureLinks connect consecutive frames and tighten monotonically under constraints', () => {
  const seq = generatePresetVideoSequence('courtyard');

  const rawRes = runAggregationPipeline(seq, {
    mode: 'raw_sharp',
    depthTemporalWeight: 0,
    globalConstraintWeight: 0,
    mergeRadiusMeters: 0,
    maxActiveFrame: 15,
  });

  const dpRes = runAggregationPipeline(seq, {
    mode: 'depth_pro_2l',
    depthTemporalWeight: 0.9,
    globalConstraintWeight: 0,
    mergeRadiusMeters: 0,
    maxActiveFrame: 15,
  });

  const fusedRes = runAggregationPipeline(seq, {
    mode: 'global_fused',
    depthTemporalWeight: 0.9,
    globalConstraintWeight: 0.9,
    mergeRadiusMeters: 0.1,
    maxActiveFrame: 15,
  });

  assert.ok(rawRes.consecutiveFeatureLinks.length > 20);
  assert.equal(rawRes.consecutiveFeatureLinks.length, fusedRes.consecutiveFeatureLinks.length);
  for (const link of fusedRes.consecutiveFeatureLinks) {
    assert.equal(link.frameTo, link.frameFrom + 1);
    assert.ok(link.gapMeters >= 0);
  }
  assert.ok(fusedRes.featureAnchors.length > 3);
  assert.ok(dpRes.metrics.meanFeatureGapMm < rawRes.metrics.meanFeatureGapMm);
  assert.ok(fusedRes.metrics.meanFeatureGapMm < dpRes.metrics.meanFeatureGapMm);

  const avgRawPull =
    rawRes.consecutiveFeatureLinks.reduce((a, l) => a + l.pullMeters, 0) /
    rawRes.consecutiveFeatureLinks.length;
  const avgFusedPull =
    fusedRes.consecutiveFeatureLinks.reduce((a, l) => a + l.pullMeters, 0) /
    fusedRes.consecutiveFeatureLinks.length;
  assert.equal(avgRawPull, 0);
  assert.ok(avgFusedPull > 0.1);
});


test('demo aggregation refuses unmarked, imported and measured sequences at runtime', () => {
  const synthetic = generatePresetVideoSequence('courtyard');
  const params = {
    mode: 'global_fused' as const,
    depthTemporalWeight: 0.9,
    globalConstraintWeight: 0.9,
    mergeRadiusMeters: 0.1,
    maxActiveFrame: 1,
  };
  for (const provenance of [undefined, 'imported', 'measured']) {
    const external = { ...synthetic, provenance } as unknown as VideoSequence;
    assert.throws(() => runAggregationPipeline(external, params), /only synthetic sequences/);
  }
});

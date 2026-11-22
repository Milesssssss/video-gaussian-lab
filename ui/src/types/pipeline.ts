/** Presentation types for the synthetic UI demo. These are not Phase 1 data contracts. */
export type PipelineMode = 'raw_sharp' | 'depth_pro_2l' | 'global_fused';

export type ColorMode = 'rgb' | 'depth' | 'error';

export type VideoPresetId = 'courtyard' | 'workbench' | 'greenhouse' | 'custom';

export type ActiveTab = 'viewport' | 'inspector' | 'benchmarks' | 'blueprint';

export type CameraPreset = 'free' | 'top' | 'frame' | 'reset';

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

/** Simplified billboard parameters; not a lossless physical Gaussian representation. */
export interface GaussianSplat {
  id: number;
  position: [number, number, number];
  scale: [number, number, number];
  color: [number, number, number];
  opacity: number;
  frameIndex: number;
  depthMeters: number;
  residualError: number;
  mergedCount: number;
}

export interface CameraPose {
  frameIndex: number;
  timestampSec: number;
  position: [number, number, number];
  lookAt: [number, number, number];
  focalPx: number;
  scaleFactor: number;
  driftMm: number;
}

export interface FeatureMatch2D {
  id: number;
  u1: number;
  v1: number;
  u2: number;
  v2: number;
  landmark3D: [number, number, number];
  reprojErrorPx: number;
  inlier: boolean;
}

export interface FrameData {
  frameIndex: number;
  timestampSec: number;
  width: number;
  height: number;
  rgbPixels: Uint8ClampedArray;
  depthMapMeters: Float32Array;
  minDepth: number;
  maxDepth: number;
  meanDepth: number;
  estimatedFocalPx: number;
  rawPose: CameraPose;
  /** Known pose of the generated synthetic scene, never a predicted pose labelled as truth. */
  gtPose: CameraPose;
  rawSplats: GaussianSplat[];
  matchesToNext: FeatureMatch2D[];
}

export interface PipelineParams {
  mode: PipelineMode;
  depthTemporalWeight: number; // Synthetic scale-noise interpolation strength, 0..1
  globalConstraintWeight: number; // GT-guided synthetic pose interpolation strength, 0..1; no BA
  mergeRadiusMeters: number; // Euclidean proximity threshold for illustrative merging
  maxActiveFrame: number; // 0 to frameCount - 1 (for incremental playback)
}

/** Generated demo curves; not research measurements or model comparisons. */
export interface FrameMetricPoint {
  frameIndex: number;
  rawAteMm: number;
  depthProAteMm: number;
  fusedAteMm: number;
  rawJitterCm: number;
  depthProJitterCm: number;
  fusedJitterCm: number;
  rawReprojPx: number;
  fusedReprojPx: number;
  rawSplatCount: number;
  fusedSplatCount: number;
}

export interface ConsecutiveFeatureLink {
  featureId: number;
  frameFrom: number;
  frameTo: number;
  from: [number, number, number];
  to: [number, number, number];
  rawFrom: [number, number, number];
  rawTo: [number, number, number];
  anchorPos: [number, number, number];
  gapMeters: number;
  pullMeters: number;
  inlier: boolean;
}

export interface FeatureAnchorNode {
  featureId: number;
  position: [number, number, number];
  observationCount: number;
  meanGapMeters: number;
  inlierRatio: number;
}

export interface AggregationResult {
  splats: GaussianSplat[];
  poses: CameraPose[];
  featureTracks: Array<{
    from: [number, number, number];
    to: [number, number, number];
    errorMeters: number;
    inlier: boolean;
  }>;
  consecutiveFeatureLinks: ConsecutiveFeatureLink[];
  featureAnchors: FeatureAnchorNode[];
  /** Synthetic scene statistics and generated error illustrations, not measured experiment metrics. */
  metrics: {
    activeFrames: number;
    rawSplatCount: number;
    outputSplatCount: number;
    compressionRatio: number;
    meanAteMm: number;
    meanReprojPx: number;
    depthJitterCm: number;
    meanFeatureGapMm: number;
    inlierLinkCount: number;
  };
  perFrameSeries: FrameMetricPoint[];
}

export interface VideoSequence {
  provenance: 'synthetic';
  id: VideoPresetId;
  title: string;
  subtitle: string;
  fps: number;
  durationSec: number;
  resolutionLabel: string;
  frames: FrameData[];
}

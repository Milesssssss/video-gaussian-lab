/** Numeric helpers used only by the synthetic UI demo; no model or video pipeline. */
import { F32C1_t, F32C2_t, imgproc, matrix_t } from '@webarkit/jsfeat-next';
import { determinant, Matrix, SingularValueDecomposition } from 'ml-matrix';

export interface UmeyamaSim3Result {
  scale: number;
  rotation: number[][];
  translation: [number, number, number];
  rmse: number;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/** Closed-form point alignment with a demo scale bound of [0.25, 4]. Not bundle adjustment. */
export function solveUmeyamaSim3(
  sourcePts: [number, number, number][],
  targetPts: [number, number, number][],
): UmeyamaSim3Result {
  const n = Math.min(sourcePts.length, targetPts.length);
  if (n === 0) {
    return {
      scale: 1,
      rotation: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      translation: [0, 0, 0],
      rmse: 0,
    };
  }

  const srcMat = new Matrix(sourcePts.slice(0, n));
  const dstMat = new Matrix(targetPts.slice(0, n));

  const muSrc = srcMat.mean('column');
  const muDst = dstMat.mean('column');

  const srcCentered = srcMat.clone().subRowVector(muSrc);
  const dstCentered = dstMat.clone().subRowVector(muDst);

  // Covariance Sigma = (1/n) * dstCentered^T * srcCentered  (3x3)
  const sigma = dstCentered.transpose().mmul(srcCentered).div(n);

  // Source variance = (1/n) * sum ||src_i - mu_src||^2
  let varSrc = 0;
  for (let i = 0; i < n; i++) {
    varSrc +=
      srcCentered.get(i, 0) ** 2 +
      srcCentered.get(i, 1) ** 2 +
      srcCentered.get(i, 2) ** 2;
  }
  varSrc /= n;

  const svd = new SingularValueDecomposition(sigma);
  const U = svd.leftSingularVectors;
  const V = svd.rightSingularVectors;
  const S = svd.diagonal;

  const detUV = determinant(U) * determinant(V);
  const dDiag = [1, 1, detUV < 0 ? -1 : 1];
  const D = Matrix.diag(dDiag);

  const R = U.mmul(D).mmul(V.transpose());
  const traceSD = S[0] * dDiag[0] + S[1] * dDiag[1] + S[2] * dDiag[2];
  const scale = varSrc > 1e-10 ? clamp(traceSD / varSrc, 0.25, 4.0) : 1.0;

  const muSrcCol = Matrix.columnVector(muSrc);
  const muDstCol = Matrix.columnVector(muDst);
  const tVec = muDstCol.sub(R.mmul(muSrcCol).mul(scale));
  const translation: [number, number, number] = [
    tVec.get(0, 0),
    tVec.get(1, 0),
    tVec.get(2, 0),
  ];

  // Evaluate RMSE
  const aligned = srcMat.mmul(R.transpose()).mul(scale).addRowVector(translation);
  let sqErr = 0;
  for (let i = 0; i < n; i++) {
    const dx = aligned.get(i, 0) - dstMat.get(i, 0);
    const dy = aligned.get(i, 1) - dstMat.get(i, 1);
    const dz = aligned.get(i, 2) - dstMat.get(i, 2);
    sqErr += dx * dx + dy * dy + dz * dz;
  }

  return {
    scale,
    rotation: R.to2DArray(),
    translation,
    rmse: Math.sqrt(sqErr / n),
  };
}

/** Sobel gradient of a synthetic scene depth grid; no confidence calibration. */
export function computeDepthEdgeGradientJsfeat(
  depthMapMeters: Float32Array,
  w: number,
  h: number,
): Float32Array {
  const depthMat = new matrix_t(w, h, F32C1_t);
  depthMat.data.set(depthMapMeters);
  const sobelMat = new matrix_t(w, h, F32C2_t);
  imgproc.sobel_derivatives(depthMat, sobelMat);

  const edgeGradient = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    // Normalize Sobel 3x3 factor (sum of positive weights = 4) to match finite-difference meter scale
    const gx = sobelMat.data[i * 2] * 0.25;
    const gy = sobelMat.data[i * 2 + 1] * 0.25;
    edgeGradient[i] = Math.hypot(gx, gy);
  }
  return edgeGradient;
}

import test from 'node:test';
import assert from 'node:assert/strict';
import { exportDemoPointCloud } from './exportDemo';
import { generatePresetVideoSequence } from '../demo/gaussianAggregation';
import type { VideoSequence } from '../types/pipeline';

test('demo export declares point-cloud provenance and does not invent Gaussian rotations', () => {
  const sequence = generatePresetVideoSequence('workbench');
  const points = sequence.frames[0].rawSplats.slice(0, 2);
  const result = exportDemoPointCloud(sequence, points);
  assert.match(result, /synthetic_demo_not_a_reconstruction_result/);
  assert.match(result, /element vertex 2/);
  assert.doesNotMatch(result, /property float rot_/);
  assert.equal(result.trimEnd().split('end_header\n')[1].split('\n').length, 2);
  assert.throws(() => exportDemoPointCloud({ ...sequence, provenance: 'local-video-preview' } as unknown as VideoSequence, points));
  assert.throws(() => exportDemoPointCloud(sequence, [{ ...points[0], position: [NaN, 0, 0] }]));
});

import test from 'node:test';
import assert from 'node:assert/strict';
import { MAX_VIDEO_BYTES, previewTimestamps, validateVideoFile } from './videoPreview';

test('local preview validates type, empty and oversized inputs before decoding', () => {
  assert.doesNotThrow(() => validateVideoFile({ name: '录制.mp4', size: 1024, type: 'video/mp4' }));
  assert.doesNotThrow(() => validateVideoFile({ name: 'CAPTURE.MOV', size: 1024, type: '' }));
  for (const file of [
    { name: 'clip.mp4', size: 0, type: 'video/mp4' },
    { name: 'clip.mp4', size: MAX_VIDEO_BYTES + 1, type: 'video/mp4' },
    { name: 'file.json', size: 10, type: 'application/json' },
    { name: 'file.mp4', size: 10, type: 'text/plain' },
  ]) assert.throws(() => validateVideoFile(file));
});
test('sampling avoids the undecodable end timestamp, including very short clips', () => {
  for (const duration of [0.001, 1, 1200]) {
    const points = previewTimestamps(duration);
    assert.equal(points.length, 6);
    assert.ok(points.every((point, index) => point > 0 && point < duration && (index === 0 || point > points[index - 1])));
  }
  for (const duration of [0, -1, Infinity, NaN]) assert.throws(() => previewTimestamps(duration));
  assert.throws(() => previewTimestamps(5, 0));
  assert.throws(() => previewTimestamps(5, 100));
});

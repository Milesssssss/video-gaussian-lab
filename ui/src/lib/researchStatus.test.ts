import assert from 'node:assert/strict';
import test from 'node:test';
import { getResearchStatus, phase1Protocol, type ProtocolRegistration } from './researchStatus';

test('missing results remain unavailable even when execution is enabled in registration', () => {
  const candidate: ProtocolRegistration = { ...phase1Protocol, execution_enabled: true, results: null };
  const status = getResearchStatus(candidate);
  assert.equal(status.executionEnabled, true);
  assert.equal(status.resultState, 'not_provided');
  assert.equal(status.implementationAvailable, false);
});

test('G0 NA is distinct from the current runner guard requiring independent geometry', () => {
  const candidate: ProtocolRegistration = structuredClone(phase1Protocol);
  for (const key of Object.keys(candidate.entry_conditions) as Array<keyof typeof candidate.entry_conditions>) {
    candidate.entry_conditions[key] = true;
  }
  candidate.entry_conditions.independent_geometry_available = false;
  const status = getResearchStatus(candidate);
  assert.equal(status.geometryState, 'NA');
  assert.equal(status.unmetProtocolConditions.length, 0);
  assert.deepEqual(status.unmetExecutionConditions.map((condition) => condition.key), ['independent_geometry_available']);
  assert.equal(status.executionEnabled, false);
});

test('an unparsed results object is not promoted to verified metrics', () => {
  const candidate: ProtocolRegistration = { ...phase1Protocol, results: { arbitrary: 'not a validated run' } };
  assert.equal(getResearchStatus(candidate).resultState, 'payload_not_connected');
});

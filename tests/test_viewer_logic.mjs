// Layer 15 checks: the viewer's segment-to-seekbar-position math.
// Run with: node tests/test_viewer_logic.mjs

import { segmentToBarStyle } from '../viewer/logic.js';

const CHECKS = [];
function check(name, fn) { CHECKS.push([name, fn]); }

check('real detected segment (4.0-9.0s of a 13.0s video) positions correctly', () => {
  const seg = { start_s: 4.0, end_s: 9.0 };
  const { left, width } = segmentToBarStyle(seg, 13.0);
  const expectedLeft = (4.0 / 13.0) * 100;
  const expectedWidth = (5.0 / 13.0) * 100;
  assertClose(left, expectedLeft, 'left');
  assertClose(width, expectedWidth, 'width');
});

check('segment at the very start of the video (left = 0)', () => {
  const { left } = segmentToBarStyle({ start_s: 0.0, end_s: 2.0 }, 10.0);
  assertClose(left, 0, 'left');
});

check('segment ending exactly at video duration (right edge = 100%)', () => {
  const { left, width } = segmentToBarStyle({ start_s: 8.0, end_s: 10.0 }, 10.0);
  assertClose(left + width, 100, 'right edge');
});

check('very short segment gets a minimum visible width (never 0)', () => {
  const { width } = segmentToBarStyle({ start_s: 500.0, end_s: 500.1 }, 1000.0);
  if (width < 0.3) throw new Error(`expected floor of 0.3%, got ${width}`);
});

check('zero or missing duration does not crash, returns zero-width', () => {
  const a = segmentToBarStyle({ start_s: 1.0, end_s: 2.0 }, 0);
  const b = segmentToBarStyle({ start_s: 1.0, end_s: 2.0 }, undefined);
  if (a.left !== 0 || a.width !== 0) throw new Error('expected zeroed output for duration=0');
  if (b.left !== 0 || b.width !== 0) throw new Error('expected zeroed output for duration=undefined');
});

function assertClose(actual, expected, label, tol = 0.01) {
  if (Math.abs(actual - expected) > tol) {
    throw new Error(`${label}: expected ~${expected.toFixed(2)}, got ${actual.toFixed(2)}`);
  }
}

let failures = 0;
for (const [name, fn] of CHECKS) {
  try {
    fn();
    console.log(`PASS  ${name}`);
  } catch (err) {
    failures++;
    console.log(`FAIL  ${name}\n        ${err.message}`);
  }
}
console.log(`\n${CHECKS.length - failures}/${CHECKS.length} checks passed`);
process.exit(failures ? 1 : 0);
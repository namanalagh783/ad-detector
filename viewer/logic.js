// Pure logic only -- no DOM access -- so it can be unit-tested with plain
// Node (see tests/test_viewer_logic.mjs) and also loaded directly by the
// browser via a <script> tag. A wrong percentage here silently misplaces
// every segment on the seek bar, so this is the one piece of viewer
// logic worth actually verifying rather than eyeballing.

const AD_TYPE_COLORS = {
  preroll: '#ff6b6b', midroll_sponsor_read: '#ffd93d', product_placement: '#74c7ec',
  self_promo: '#a29bfe', affiliate: '#55efc4', platform_inserted: '#fd79a8',
  bumper: '#fab1a0', other: '#b2bec3',
};

function segmentToBarStyle(seg, durationS) {
  if (!durationS || durationS <= 0) return { left: 0, width: 0 };
  const left = (seg.start_s / durationS) * 100;
  const width = Math.max(((seg.end_s - seg.start_s) / durationS) * 100, 0.3);
  return { left, width };
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { segmentToBarStyle, AD_TYPE_COLORS };
}
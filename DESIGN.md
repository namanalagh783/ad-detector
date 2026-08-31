# DESIGN.md — Ad Segment Detection Pipeline

## AI tool use disclosure

*(TODO — fill in honestly before submitting. Suggested framing, edit as
true: architecture and every module were built incrementally with Claude
across 16 tested layers, each verified before the next was built on top
of it; several real bugs were found and fixed along the way (see Section
4). The ad taxonomy and the 8 ambiguity-pack rulings below are my own
judgment, reasoned through in conversation but decided by me. The cost/
scale projections in Sections 5-7 were AI-drafted from real data and
constraints found during the build, then reviewed by me.)*

## 1. Definition of an ad, and taxonomy

An ad, for this pipeline, is a segment of **bounded duration, shorter
than the full video**, that either (a) promotes a third-party brand,
product, or service, typically in exchange for compensation, or (b)
directly asks the audience to financially support the creator/channel
(self-promotion). Content that spans the *entire* video regardless of
promotional character is deliberately not treated as a segment — it
fails the "bounded, shorter than the video" shape before the question of
intent even matters (see ambiguity items 2 and 7 below, which is exactly
why they're ruled differently despite looking similar on the surface).
Content whose promotional nature can't be reliably recognized by the
current signal stack is named as an explicit blind spot, not guessed at
or silently ignored (item 4).

How the 8 `ad_type` values map onto this definition:

- `midroll_sponsor_read` — the primary long-form type, ASR-driven (a
  scripted spoken sponsor mention, with or without a CTA).
- `bumper` — a short animated/graphic ad card, OCR-driven.
- `product_placement` — VLM-driven, used when a frame is visually
  ad-related but neither ASR nor OCR flagged anything.
- `self_promo` — the creator asking for direct support of themselves/the
  channel (e.g. "join my Patreon"). Explicitly NOT used for incidental,
  whole-video brand presence (item 2) — see Section 2, item 1 for why
  the line is drawn where it is.
- `affiliate` — CTA-style link/code mentions tied to a specific product.
- `platform_inserted` — evidence of a platform-level ad insertion is
  detectable (e.g. a jump cut, a metadata marker) even when the ad
  content itself isn't in the downloaded bytes (item 5).
- `preroll` — an ad at the very start of a video, before content begins.
- `other` — fallback for anything that's clearly promotional but doesn't
  fit the above.

## 2. The ambiguity pack

1. **Patreon shoutout.** A bare "like and subscribe" is too universal to
   ever count as an ad — nearly every video has one, and treating it as
   an ad would make the label meaningless. But "join my Patreon" is a
   direct, recurring-payment call-to-action aimed at monetizing the
   audience specifically — closer to genuine self-promotion than idle
   chat. Tagged `self_promo`, not excluded entirely.

2. **Host wears their own merch hoodie for the entire video.** Not
   flagged. An ad segment requires a bounded start and end shorter than
   the video itself; something present for the full runtime fails that
   shape before the promotional question is even relevant. This is
   ambient wardrobe, not a segment — distinct from item 7 below, where
   the *entire* Reel was deliberately built to be the ad.

3. **A 1.4s animated "sponsored by" bumper — what does this do to
   sampling?** Uniform sampling at any reasonable rate can straddle and
   miss a sub-2-second event entirely. The fix implemented in
   `sampling.py` is to densify sampling specifically around detected
   scene cuts, rather than raising the global sampling rate everywhere —
   the latter would waste budget on long static stretches for no benefit
   while still not guaranteeing coverage of a very short event.

4. **A movie review that plays 40s of the official trailer.** Licensed
   trailer content is arguably promotional — the studio wants it seen —
   but reliably recognizing "this specific clip is official marketing"
   requires brand/franchise recognition that the current signal stack
   doesn't attempt. Named as a known blind spot rather than silently
   ignored: the system will not detect this case, and that limitation is
   stated here on purpose rather than discovered by a reviewer.

5. **A platform pre-roll never in the downloaded stream at all.** If an
   ad isn't present in the downloaded bytes, no signal — audio, visual,
   or otherwise — can detect it. This is out of scope by construction,
   not a gap being pretended away. `platform_inserted` exists for the
   related but different case where *evidence* of an insertion (a jump
   cut, a metadata hint) is detectable even when the ad content itself
   isn't recoverable.

6. **A 90s sponsor read where the visual never changes — what signal do
   you have?** Audio is the only signal that exists for that 90 seconds
   once the visual stops changing — OCR and VLM have nothing new to look
   at after the first frame. This is the direct justification for why
   ASR is the *primary* detector for long-form video in this pipeline,
   not a secondary/backup signal.

7. **A Reel that's an ad from frame one to the last frame.**
   `start_s = 0`, `end_s = duration` — one segment spanning the entire
   clip. This looks structurally identical to item 2 (something spanning
   the whole video) but is ruled oppositely on purpose: a hoodie is
   incidental brand presence nobody scripted as an ad, while a Reel built
   entirely as sponsored content *is*, by construction and intent,
   wall-to-wall promotional material.

8. **Two sponsors back to back, with no gap.** One segment. With zero
   time gap between them, there's nothing to split on without a
   secondary signal — e.g. the brand name changing mid-window — that the
   current pipeline doesn't use for this purpose. This matches
   `fusion.py`'s existing `MERGE_GAP_S` behavior exactly: candidate
   windows within the merge threshold combine into one segment, and a
   gap of zero is well within that threshold. Splitting on the mere
   absence of a pause would be arbitrary, not principled — a real fix
   would need real evidence of a boundary, which isn't available here.

## 3. Architecture

The pipeline treats the three input formats differently by design, not
uniformly, because the strongest signal genuinely differs between them:

- **Long-form YouTube (8-60 min):** audio carries most of the signal.
  Sponsor reads are scripted speech ("sponsored by," "use code," "%
  off"), and the visual on screen frequently doesn't change at all
  during one — a frame-only pipeline has zero signal for that case. So
  transcript analysis (`asr.py` + `signals.py`) is the primary detector
  here; frames are corroborating evidence, not the primary signal.
- **Shorts / Reels (5-90s):** inverted. Dense visual signal, on-screen
  text and numbers, often little or no narration, and sometimes the
  *entire* clip is the ad with no internal boundary at all (ambiguity
  item 7). OCR and VLM frame classification carry more weight here.
- **Live streams:** not yet implemented (see Section 7).

**Pipeline stages, in order** (`pipeline.py` orchestrates all of these):

1. **Ingestion** (`ingest.py`) — local files, manually-acquired Reels
   (per the assignment's explicit acquisition rule), or real YouTube
   download via yt-dlp. Platform/kind detection from URL shape, falling
   back to yt-dlp's own metadata (`is_live`/`was_live`) when the URL
   shape alone is ambiguous (a plain `watch?v=` URL could be either).
2. **Scene-cut detection** (`scenes.py`) — cheap, ffmpeg-based, no ML.
   Exists purely to tell sampling *where* to look, not to detect ads
   itself.
3. **Adaptive sampling** (`sampling.py`) — a sparse uniform "backbone"
   (denser for Shorts than long-form) plus densified sampling around
   every scene cut. This is the direct, concrete answer to ambiguity
   item 3: a uniform sampling rate would frequently straddle and miss a
   sub-2-second visual event like an animated bumper; densifying
   sampling specifically around detected cuts catches it without
   raising the global frame rate everywhere (which would blow the cost
   budget for no benefit on long static stretches).
4. **Frame extraction + OCR** (`frames.py`, `ocr.py`) — pulls the actual
   pixels at each sampled timestamp, then reads on-screen text and flags
   ad-indicator phrases ("sponsored," "affiliate," "% off," etc.). Cheap
   and local, so it runs on every sampled frame, not just candidates.
5. **Audio extraction + ASR** (`audio.py`, `asr.py`) — faster-whisper,
   local and free (no per-call cost), which matters for the assignment's
   $10 total-spend cap.
6. **Sponsor-language spotting** (`signals.py`) — scans the transcript
   for two separate kinds of phrase: a "launch" phrase ("sponsored by")
   and a "call-to-action" phrase ("use code," "% off"). Kept separate
   rather than collapsed into one "is this sponsor-y" flag, because
   fusion weighs them differently and a segment can have either, both,
   or neither.
7. **VLM classification** (`vlm.py`) — Gemini vision, called ONLY on a
   pre-filtered, cost-capped subset of frames: those near a transcript
   hit, those OCR already flagged, or (fallback, only if neither cheaper
   signal found anything at all) frames near a scene cut. This
   candidate-gating, implemented in `pipeline.py`'s
   `_select_vlm_candidates`, is the actual mechanism keeping
   `stats.estimated_cost_usd` and `model_calls` bounded — never every
   sampled frame gets a paid call.
8. **Fusion** (`fusion.py`) — merges all signals into final segments.
   Candidate windows within a 2-second gap of each other merge into one
   segment (the direct answer to ambiguity item 8: no gap means nothing
   to split on). Boundaries snap to the nearest real scene cut within a
   1-second tolerance, cleaning up fuzzy signal-derived timestamps into
   crisp edges. `ad_type` is assigned by an explicit priority ruleset,
   not a black-box classifier, specifically so a reviewer can see
   exactly why a segment got its label.

## 4. What didn't work

Kept here honestly rather than quietly fixed and hidden, since the
assignment explicitly asks for this:

- **ffmpeg's commonly-cited scene-detection threshold (0.3) is not a
  safe default.** Built a synthetic fixture with two known hard cuts and
  found that a cut between two similarly-dark colors scored well under
  0.3, while a cut into a bright color scored far above it — both are
  equally real cuts, but the scene score scales with color/luminance
  delta, not with "is this a cut." Lowered the default to 0.08 after
  finding this empirically; it's a real precision/recall trade-off (a
  lower threshold also fires more on noisy real footage like camera
  shake), not a solved problem.
- **A boundary-snapping bug that inverted the intended logic.** The
  first version of `_snap_to_scene_cuts` only credited `scene_cut` as a
  contributing signal when snapping actually *moved* a boundary. But a
  raw signal boundary landing exactly ON a real scene cut — the
  strongest possible agreement — produced no movement, and so was
  silently read as "no evidence" instead of the best evidence available.
  Fixed by returning an explicit `had_scene_cut_support` boolean instead
  of inferring it from whether a value changed.
- **The `bumper` ad_type was nearly unreachable.** The original priority
  ruleset only classified a segment as `bumper` when OCR found
  ad-indicator text AND the VLM model disagreed that it was ad-visual.
  In practice, a vision model correctly agreeing "yes, this is an ad" is
  the *normal* case for an obvious bumper card, not an edge case — so
  real bumpers were almost always falling through to `product_placement`
  instead. Caught this against a real, non-mocked pipeline run (not an
  offline test), not by inspection. Fixed by letting OCR's ad-indicator
  text take priority regardless of what VLM separately concludes.
- **A silent font-fallback bug in test-fixture generation** (not
  production code): a synthetic fixture used a font name
  (`arial.ttf`) that doesn't exist on the machine it was first built on,
  and PIL silently fell back to its tiny default font. The resulting
  bumper card was visually correct but had unreadably small text, so
  OCR found nothing — which looked exactly like a real detection bug
  until the frame was actually viewed. Fixed by trying several common
  font names and printing an explicit warning on fallback, instead of
  failing silently.

## 5. Cost and latency

*(TODO — needs real runs against the full 5-video test set, not just the
synthetic fixture below, before this section can be considered complete.
What's here is real data from one full non-mocked run, not a placeholder.)*

One complete real run (13-second synthetic fixture, `pipeline_fixture.mp4`,
1 detected segment, 5 VLM candidate frames):

| Stage | Time |
|---|---|
| Ingest | 2.2s |
| Scene-cut detection | 1.9s |
| Sampling + frame extraction | 0.5s |
| OCR | 5.9s |
| Audio extraction | 0.1s |
| ASR (faster-whisper, CPU, "small") | 16.4s |
| **VLM classification (5 frames)** | **290.2s** |
| Fusion | <0.1s |
| **Total** | **317.2s** |

**VLM classification is 91% of total wall-clock time**, and it's not
compute cost — it's provider-side latency and reliability. Individual
Gemini free-tier calls ranged from 5s to 22s, with two outright `503
(high demand)` errors observed in a follow-up timing test. This isn't a
simple rate-limit pattern (which would look like several fast calls then
a hard block); it looks like the free tier itself being genuinely
overloaded server-side. Estimated cost for this run: $0 (free tier).

**What this means for "what would you cut to get 10x cheaper":** the
answer isn't algorithmic — it's provider choice. A paid tier of the same
model family would very likely eliminate most of this latency at a cost
that's still trivial per frame (a handful of cents for the entire 5-video
test set, well under the $10 cap) — the free tier is optimizing for $0
at the cost of latency and reliability, which is a real, honest
trade-off to name rather than something to solve by writing more code.

## 6. What breaks at 1000 videos/day

*(Drafted from real constraints observed during the build — review and
edit as needed, this is a projection, not measured.)*

- **VLM throughput is the hard ceiling.** Even a conservative 10-20
  candidate frames/video average means 10,000-20,000 VLM calls/day.
  Gemini's free tier (roughly 10-30 requests/minute depending on model)
  would take many hours just to clear one day's queue serially, before
  accounting for the 503 flakiness already observed at low volume. A
  paid tier with real throughput guarantees, plus a proper
  concurrency-limited worker pool with retry/backoff, is not optional at
  this scale.
- **CPU-based Whisper doesn't parallelize well on one machine.** Fine
  for one video at a time; at ~42 videos/hour average (1000/day, and
  real traffic is bursty, not uniform), would need either GPU-
  accelerated transcription or a horizontally-scaled worker pool.
- **Ingestion, scene detection, frame extraction, and OCR are
  embarrassingly parallel** across videos (each video is independent,
  CPU/IO-bound, no shared state) — these scale fine by just adding
  worker machines.
- **Storage would balloon** if downloaded videos and extracted frames
  aren't cleaned up after processing — only the output JSON and a small
  set of evidence frame thumbnails need to persist long-term.
- **Architecture implication:** move from "one process runs a video
  start-to-finish" to a queue-based pipeline where the slow stage (VLM)
  has its own rate-limited worker pool and doesn't block the cheap
  stages for other videos behind it.

## 7. What I'd build next with two more weeks

*(Drafted — review and edit)*

1. Live-stream windowing (Tier 2) — not yet implemented at all.
2. Retry/backoff (and possibly a fallback provider) for VLM calls, given
   the real 503 flakiness found in Section 5.
3. Confidence calibration — current confidence is a heuristic weighted
   sum (see `fusion.py`'s `_confidence`), never checked against whether
   e.g. 0.8-confidence predictions are actually right ~80% of the time.
4. A richer, platform-aware OCR ad-indicator lexicon (e.g. Instagram's
   specific "Paid partnership with X" label format).
5. Hungarian-algorithm matching for eval instead of greedy-by-IoU, once
   videos have many overlapping-ish candidate segments where greedy
   matching's simplification starts to matter.
6. Brand-name capitalization cleanup (`_extract_brand`'s `.title()` call
   currently turns "acme vpn" into "Acme Vpn," not "Acme VPN" — cosmetic,
   known, not fixed).
7. GPU-accelerated Whisper for faster/larger-model transcription.
8. Streaming output over SSE/WebSocket (Tier 3), once Tier 1/2 are solid.
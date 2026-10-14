# Hyperframes Composition Brief: gate-calibrated

## Objective
Create a short launch-style brag video for gate-calibrated (polished, honest research film).

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape 1920x1080, 30 fps
- Duration: 24 s

## Source Material
- Project root: `C:\Code Main\gate-calibrated`
- Primary files read: README.md, DESIGN.md, results/report.md, results/curve.json, results/calibration.json, results/collect/rows.jsonl, assets/logo/*.svg
- Product name: gate-calibrated
- Strongest claim (true, from the data): the agent stated confidence 1.00 on actions that were wrong; the calibrated gate held its risk targets but did no better than asking every time.
- Key visual to recreate: the barrier mark (assets/logo/build_logo.py geometry) and the measured curve (results/curve.json).
- Copy that must appear verbatim:
  - It changed one item. The user asked for two.
  - A barrier between an agent and its writes.
  - Risk held at every alpha.
  - Per approval, asking every time did as well.
  - Open and reproducible.
  - Gjusev/gate-calibrated

## Creative Direction
- Tone preset: polished
- Creative direction: a quiet engineering film about a barrier that checks every write
- Angle: see brag-plan.md
- Hook: the typed confident mistake (#W3260419, confidence 1.00, 5 of 5 agree)
- Outro: logo with one arm lift, repo name
- Avoid: generic SaaS language, abstract filler, gradients, glows, grid backgrounds, AI icons

## Visual Identity
- Background #F4F1EA, text #16181D, muted #5B5F66, line #D8D3C7, accent #E8A317 (ask state only)
- Display font Overpass (local woff2, variable 100-900), code font Overpass Mono (local woff2)

## Storyboard
Contract: brag-plan.md. Scenes: 1 confident mistake 0-5.0 / 2 the gate 5.0-8.5 / 3 act or ask 8.5-16.0 / 4 what we measured 16.0-20.5 / 5 outro 20.5-24.0.

## Audio
- Role: sparse accents, no music (bundled music license unverified for a public repo; SFX are Kenney CC0)
- SFX (warm, low HF risk): interface/click_003 for thinned key ticks; impact/impactSoft_medium_001 for arm stops; interface/bong_001 for the amber lamp; impact/impactSoft_heavy_003 for the verdict line
- Audio files copied into composition/assets/sfx/

## Hyperframes Instructions
Standalone index.html, one paused GSAP timeline, local GSAP, local fonts, check before render.

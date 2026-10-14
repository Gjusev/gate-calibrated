# DESIGN.md: gate-calibrated identity (logo, video, README)

Direction chosen by Youssef on 2026-10-14: the barrier arm concept, antislop applied during the work.

## Concept card

```
Idea:        before an agent writes, a barrier arm decides: it lifts by itself, or it stops and waits for a person.
Metaphor:    a level-crossing / car-park barrier. Mechanical, legible from a distance, two states.
Signature:   the arm lifting (act) against the arm holding with the amber lamp lit (ask).
Emotion:     controlled and calm. Something is being checked; nothing is panicking.
Motion verb: lift like a barrier: slow start, firm stop, a small settle at the end of travel.
Screenshot:  the arm held level, amber lamp on, wordmark beside it.
```

Design read: an open-source research tool for ML and agent engineers, in a road-signage language, dial ENERGY 2 / RHYTHM 2 / MOTION 2.

## Palette (2 core + 1 accent)

| role | hex | use | reason |
|---|---|---|---|
| paper | `#F4F1EA` | light ground | warm off-white of a printed sign, not a sterile #fff |
| ink | `#16181D` | marks, text, arm stripes | near-black of road paint |
| muted | `#5B5F66` | secondary text | 5.7:1 on paper, passes AA |
| line | `#D8D3C7` | hairlines, the post base | quiet structure only |
| signal | `#E8A317` | the waiting lamp, the one highlight | amber is the colour of "wait"; used only where the gate asks |

Dark variant swaps paper and ink (`#16181D` ground, `#F4F1EA` marks). Amber never carries text on paper (1.9:1, fails); on ink it is 8.2:1. Muted on the dark ground is `#9A9DA3` (6.5:1).

## Type

- Overpass (Google Fonts, OFL): the open-source interpretation of Highway Gothic, the US road-sign face. Picked because the concept is a road barrier: the wordmark should read like signage.
- Overpass Mono for code, tool names and numbers in the video, because the product's raw material is tool calls and their arguments.

## Shape and motion

- Radius: 2 px on the post and the lamp housing, none elsewhere. A barrier is a machine, not a pill.
- Arm: alternating ink and paper stripes, the one decorative motif, repeated in the video as a divider.
- Eases: one family, `cubic-bezier(.6,0,.2,1)` for travel, a 6 % overshoot settle at the end of travel. Lifts take 0.9 s, the hold before a lift 0.4 s.
- Reduced motion: the arm rests level with the lamp lit (the ask state), no animation.

## Reasons (R-31)

- Barrier arm: the product is literally a gate between a proposed action and its execution.
- Amber only on ask: the accent marks the moment a human is pulled in, so it has to mean one thing.
- No gradients, glows, grids or AI icons: none of them say anything about a barrier.

"""Build the gate-calibrated logo SVGs (see DESIGN.md).

The wordmark is Overpass, shaped with HarfBuzz and converted to outline
paths, so the SVGs render the same everywhere without loading a font.

    uv run --with fonttools --with brotli --with uharfbuzz python assets/logo/build_logo.py

Writes, next to this file:
  logo-light.svg, logo-dark.svg                  static, arm level, lamp lit (ask)
  logo-animated-light.svg, logo-animated-dark.svg  arm lifts (act), returns, lamp lights (ask)
  mark-light.svg, mark-dark.svg                  the barrier alone, square
"""

import io
import os

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "..", "fonts", "overpass-800-normal.woff2")

THEMES = {
    "light": {"ground": "#F4F1EA", "ink": "#16181D", "muted": "#5B5F66", "line": "#D8D3C7"},
    "dark": {"ground": "#16181D", "ink": "#F4F1EA", "muted": "#9A9DA3", "line": "#3A3D44"},
}
SIGNAL = "#E8A317"
EASE = "cubic-bezier(.6,0,.2,1)"


def wordmark_paths(text, weight, size, x0, baseline):
    """Outline `text` at `weight` and font `size` px; return (svg path d, advance px)."""
    var = TTFont(FONT)
    inst = instantiateVariableFont(var, {"wght": weight})
    buf = io.BytesIO()
    inst.flavor = None
    inst.save(buf)
    data = buf.getvalue()

    face = hb.Face(data)
    font = hb.Font(face)
    upm = face.upem
    hbuf = hb.Buffer()
    hbuf.add_str(text)
    hbuf.guess_segment_properties()
    hb.shape(font, hbuf, {"kern": True, "liga": True})

    glyph_set = inst.getGlyphSet()
    order = inst.getGlyphOrder()
    scale = size / upm
    pen = SVGPathPen(glyph_set)
    x = 0
    for info, pos in zip(hbuf.glyph_infos, hbuf.glyph_positions):
        name = order[info.codepoint]
        # font units -> px, y flipped, placed at (x0 + x, baseline)
        t = (scale, 0, 0, -scale, x0 + (x + pos.x_offset) * scale, baseline - pos.y_offset * scale)
        glyph_set[name].draw(TransformPen(pen, t))
        x += pos.x_advance
    return pen.getCommands(), x * scale


def barrier(c, ox, oy, animated, lamp_lit):
    """A boom barrier in a 200x200 box: low cabinet, long striped arm pivoting
    at (56, 150), signal lamp on the cabinet left of the pivot so the lifted
    arm never covers it. Lifted to -82 deg the arm tip stays inside the box."""
    stripes = []
    for i, x in enumerate(range(74, 186, 22)):
        fill = c["ink"] if i % 2 == 0 else c["ground"]
        w = min(22, 186 - x)
        stripes.append(f'<rect x="{x}" y="141" width="{w}" height="18" fill="{fill}"/>')
    arm = (
        f'<g class="arm">'
        f'<rect x="56" y="141" width="130" height="18" fill="{c["ground"]}"/>'
        f'{"".join(stripes)}'
        f'<rect x="56" y="141" width="130" height="18" rx="1.5" fill="none" stroke="{c["ink"]}" stroke-width="3"/>'
        f"</g>"
    )
    lamp_fill = SIGNAL if lamp_lit else c["line"]
    return (
        f'<g transform="translate({ox} {oy})">'
        f'<rect x="4" y="190" width="74" height="7" rx="2" fill="{c["line"]}"/>'
        f'<rect x="12" y="134" width="56" height="58" rx="3" fill="{c["ink"]}"/>'
        f'<circle class="lamp" cx="30" cy="121" r="11" fill="{lamp_fill}" stroke="{c["ink"]}" stroke-width="3"/>'
        f"{arm}"
        f'<circle cx="56" cy="150" r="7" fill="{c["ground"]}" stroke="{c["ink"]}" stroke-width="3"/>'
        f"</g>"
    )


def animation_css(c):
    """6 s loop: level, lift (act), hold, lower, lamp lights and waits (ask)."""
    return f"""
  .arm {{ transform-box: view-box; transform-origin: 62px 150px; animation: arm 6s {EASE} infinite; }}
  .lamp {{ animation: lamp 6s steps(1, end) infinite; }}
  @keyframes arm {{
    0%, 10% {{ transform: rotate(0deg); }}
    25% {{ transform: rotate(-86deg); }}
    28% {{ transform: rotate(-82deg); }}
    44% {{ transform: rotate(-82deg); }}
    58% {{ transform: rotate(1.5deg); }}
    61%, 100% {{ transform: rotate(0deg); }}
  }}
  @keyframes lamp {{
    0%, 61% {{ fill: {c["line"]}; }}
    62%, 72% {{ fill: {SIGNAL}; }}
    73%, 76% {{ fill: {c["line"]}; }}
    77%, 96% {{ fill: {SIGNAL}; }}
    97%, 100% {{ fill: {c["line"]}; }}
  }}
  @media (prefers-reduced-motion: reduce) {{
    .arm {{ animation: none; }}
    .lamp {{ animation: none; fill: {SIGNAL}; }}
  }}"""


def svg(width, height, body, title, style=""):
    style_el = f"<style>{style}\n</style>" if style else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-labelledby="t">'
        f'<title id="t">{title}</title>{style_el}{body}</svg>\n'
    )


def main():
    gate_d, gate_w = wordmark_paths("gate", 800, 92, 222, 166)
    rest_d, rest_w = wordmark_paths("-calibrated", 600, 92, 222 + gate_w, 166)
    width = int(222 + gate_w + rest_w + 24)
    title = "gate-calibrated: a barrier arm that lifts on its own or waits for approval"

    for theme, c in THEMES.items():
        word = (
            f'<path d="{gate_d}" fill="{c["ink"]}"/>'
            f'<path d="{rest_d}" fill="{c["muted"]}"/>'
        )
        for animated in (False, True):
            body = barrier(c, 6, 0, animated, lamp_lit=not animated) + word
            name = f'logo-{"animated-" if animated else ""}{theme}.svg'
            with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
                f.write(svg(width, 200, body, title, animation_css(c) if animated else ""))
        mark = f'<rect width="220" height="220" rx="16" fill="{c["ground"]}"/>' + barrier(c, 10, 6, False, True)
        with open(os.path.join(HERE, f"mark-{theme}.svg"), "w", encoding="utf-8") as f:
            f.write(svg(220, 220, mark, "gate-calibrated mark"))
    print(f"wrote 6 SVGs, logo width {width}px")


if __name__ == "__main__":
    main()

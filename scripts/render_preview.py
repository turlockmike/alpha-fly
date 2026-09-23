"""The social-preview card for the site (docs/assets/preview.png, 1200 x 630): the fly's neurons from
docs/model/brain.bin as a point cloud, and the page's title.  Needs Pillow and a display font.

    uv run python scripts/render_preview.py docs/model/brain.bin docs/assets/preview.png [display.ttf] [body.ttf]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

brain, out = Path(sys.argv[1]), Path(sys.argv[2])
display = sys.argv[3] if len(sys.argv) > 3 else "C:/Windows/Fonts/segoeuib.ttf"
body = sys.argv[4] if len(sys.argv) > 4 else "C:/Windows/Fonts/segoeui.ttf"

buf = brain.read_bytes()
K = int(np.frombuffer(buf, "<u4", 1)[0])
o = 4 + 4 * K
x = np.frombuffer(buf, "<i2", K, o).astype(float) / 32767
y = np.frombuffer(buf, "<i2", K, o + 2 * K).astype(float) / 32767
z = np.frombuffer(buf, "<i2", K, o + 4 * K).astype(float) / 32767
cls = np.frombuffer(buf, "u1", K, o + 6 * K)

W, H, S = 1200, 630, 2                                   # drawn at 2x and downsampled for smooth points
img = Image.new("RGB", (W * S, H * S), (241, 243, 240))
d = ImageDraw.Draw(img, "RGBA")
# the CNS, dorsal view, anterior at the top-right: a slight tilt so it reads as a specimen, not a chart
ang = -0.15
u = x * np.cos(ang) - y * np.sin(ang)
v = x * np.sin(ang) + y * np.cos(ang)
cx, cy, scale = 940 * S, 305 * S, 235 * S
order = np.argsort(z)                                   # far points first
ink, acc, teal = (19, 26, 23), (184, 35, 110), (23, 117, 102)
for i in order:
    px, py = cx + u[i] * scale, cy + v[i] * scale
    k = cls[i]
    col = teal + (150,) if k == 3 else acc + (140,) if k == 4 else ink + (70 if k == 1 else 90,)
    r = (2.6 if k >= 3 else 1.9) * S / 2
    d.ellipse([px - r, py - r, px + r, py + r], fill=col)
# a few "firing" neurons in the accent, so the card shows what the page does
rng = np.random.default_rng(3)
for i in rng.choice(K, 900, replace=False):
    px, py = cx + u[i] * scale, cy + v[i] * scale
    r = rng.uniform(2.5, 4.5) * S / 2
    d.ellipse([px - r, py - r, px + r, py + r], fill=acc + (int(rng.uniform(110, 220)),))

f_big = ImageFont.truetype(display, 66 * S)
f_small = ImageFont.truetype(body, 26 * S)
f_tag = ImageFont.truetype(body, 22 * S)
d.text((72 * S, 150 * S), "A fruit fly's brain", font=f_big, fill=ink)
d.text((72 * S, 228 * S), "would like a game.", font=f_big, fill=acc)
d.multiline_text((74 * S, 356 * S), "166,700 neurons and 10.5 million synapses,\nwired as in the male Drosophila connectome,\ntaught chess by playing itself.\nPlay it in your browser.",
                 font=f_small, fill=(75, 87, 83), spacing=10 * S)
d.ellipse([72 * S, 62 * S, 92 * S, 82 * S], fill=acc)
d.text((104 * S, 58 * S), "alpha-fly", font=ImageFont.truetype(display, 30 * S), fill=ink)
d.text((72 * S, 560 * S), "turlockmike.github.io/alpha-fly", font=f_tag, fill=(123, 135, 131))
img = img.resize((W, H), Image.LANCZOS)
out.parent.mkdir(parents=True, exist_ok=True)
img.save(out, optimize=True)
print(out, f"{out.stat().st_size / 1024:.0f} KB")

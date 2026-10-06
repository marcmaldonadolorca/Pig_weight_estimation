from PIL import Image, ImageDraw, ImageFont, ImageChops
import numpy as np

IMG = "reports/informe_final/latex/img"   # run from the repository root
F  = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

CELL_W, CELL_H = 520, 400
CAP_H, PAD     = 86, 18
BG    = (255, 255, 255)
CARD  = (247, 248, 250)
INK   = (17, 20, 24)
MUTE  = (110, 118, 128)
LINE  = (226, 230, 235)
ACC   = (37, 99, 190)      # mask silhouette

def load(fn):
    im = Image.open(f"{IMG}/{fn}")
    if im.mode == "RGBA":
        flat = Image.new("RGB", im.size, (255, 255, 255))
        flat.paste(im, mask=im.split()[3])
        im = flat
    return im.convert("RGB")

def autocrop(im, tol=12):
    bg = Image.new("RGB", im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg).convert("L").point(lambda p: 255 if p > tol else 0)
    box = diff.getbbox()
    return im.crop(box) if box else im

def silhouette(im, colour):
    """Grayscale mask -> coloured silhouette on card background."""
    a = np.asarray(im.convert("L"))
    fg = a < 128                                   # pig is dark in the source mask
    out = np.empty(a.shape + (3,), np.uint8)
    out[...] = CARD
    out[fg] = colour
    return Image.fromarray(out)

def key_grey(im, tol=18):
    """Replace a flat neutral viewer background with the card colour."""
    a = np.asarray(im).astype(int)
    ref = a[0, 0]
    bg = (np.abs(a - ref).max(axis=2) <= tol)
    a[bg] = CARD
    return Image.fromarray(a.astype(np.uint8))

def key_blue(im):
    """Replace the 3D viewer's blue gradient; the subject is warm-toned."""
    a = np.asarray(im).astype(int)
    bg = a[..., 2] > a[..., 0] + 18                # blue dominates red
    a[bg] = CARD
    return Image.fromarray(a.astype(np.uint8))

def fit(im, w, h):
    im = im.copy()
    im.thumbnail((w - 24, h - 24), Image.LANCZOS)
    cell = Image.new("RGB", (w, h), CARD)
    cell.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return cell

p1 = autocrop(load("YOLOBOX.png"))
p2 = autocrop(silhouette(load("mask.jpg"), ACC), tol=8)
p3 = autocrop(key_grey(load("pcdbien.png")), tol=8)
p4 = autocrop(key_blue(load("mesh.png")), tol=8)

PANELS = [
    (p1, "1 · Overhead IR frame", "YOLOv5 detection"),
    (p2, "2 · Semantic mask",     "U-Net · IoU 0.98"),
    (p3, "3 · Point cloud",       "Open3D, depth-coloured"),
    (p4, "4 · Surface mesh",      "Poisson reconstruction"),
]

f_title, f_sub = ImageFont.truetype(FB, 26), ImageFont.truetype(F, 22)
n = len(PANELS)
W = n * CELL_W + (n + 1) * PAD
H = CELL_H + CAP_H + 2 * PAD
strip = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(strip)

for i, (im, title, sub) in enumerate(PANELS):
    x = PAD + i * (CELL_W + PAD)
    strip.paste(fit(im, CELL_W, CELL_H), (x, PAD))
    d.rectangle([x, PAD, x + CELL_W - 1, PAD + CELL_H - 1], outline=LINE, width=2)
    ty = PAD + CELL_H + 18
    d.text((x, ty), title, font=f_title, fill=INK)
    d.text((x, ty + 34), sub, font=f_sub, fill=MUTE)

strip.save("pipeline.png", optimize=True)
print("pipeline.png", strip.size)

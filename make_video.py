"""
make_video.py -- side-by-side supplementary video: Proposed (left) vs B3 certainty-equivalent (right),
time-synchronised, with live overlays from the run logs.

Inputs : /tmp/zaman_media/P/frame_*.jpg, /tmp/zaman_media/B3/frame_*.jpg,
         runs/img_P.csv, runs/img_B3.csv   (written by carla_friction_bound.py --video)
Output : media/side_by_side_S1.mp4

    python make_video.py
"""
import glob, os, re, subprocess
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SRC = "/tmp/zaman_media"
TMP = os.path.join(SRC, "sbs")
WET = (387.0, 1070.0)                      # wet patch [m], as printed by the S1 run
os.makedirs(TMP, exist_ok=True)
os.makedirs("media", exist_ok=True)


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_TITLE, F_TXT, F_BADGE = font(26, True), font(22), font(30, True)


def frames(method):
    fs = sorted(glob.glob(f"{SRC}/{method}/frame_*.jpg"))
    ks = np.array([int(re.search(r"frame_(\d+)", f).group(1)) for f in fs])
    return fs, ks


def row_at(df, t):
    i = int(np.clip(np.searchsorted(df.t.values, t), 0, len(df) - 1))
    return df.iloc[i]


def panel(img_path, r, title, plan_label, plan_val, ended):
    im = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 0, im.width, 44], fill=(0, 0, 0, 170))
    d.text((14, 8), title, font=F_TITLE, fill=(255, 255, 255))
    lines = [f"distance  {r.s:6.0f} m",
             f"speed     {r.v:6.1f} m/s",
             f"μ estimate   {r.mu_hat:4.2f}",
             f"{plan_label}  {plan_val:4.2f}"]
    d.rectangle([10, 56, 330, 56 + 30 * len(lines) + 10], fill=(0, 0, 0, 140))
    for j, ln in enumerate(lines):
        d.text((20, 62 + 30 * j), ln, font=F_TXT, fill=(255, 255, 255))
    if WET[0] <= r.s <= WET[1] and not ended:
        d.rectangle([im.width - 230, 56, im.width - 12, 104], fill=(30, 110, 220, 210))
        d.text((im.width - 214, 62), "WET ROAD", font=F_BADGE, fill=(255, 255, 255))
    if ended:
        d.rectangle([0, im.height // 2 - 40, im.width, im.height // 2 + 40], fill=(200, 30, 30, 200))
        d.text((im.width // 2 - 110, im.height // 2 - 20), "LEFT LANE", font=F_BADGE, fill=(255, 255, 255))
    return im


def main():
    P = pd.read_csv("runs/img_P.csv"); B = pd.read_csv("runs/img_B3.csv")
    fP, kP = frames("P"); fB, kB = frames("B3")
    t_end_B = B.t.iloc[-1]
    for n, (fp, k) in enumerate(zip(fP, kP)):
        t = k * 0.01
        rp = row_at(P, t)
        jb = int(np.clip(np.searchsorted(kB, k, side="right") - 1, 0, len(fB) - 1))
        rb = row_at(B, min(t, t_end_B))
        left = panel(fp, rp, "Proposed: excitation-aware bound", "planning μ (bound)", rp.mu_lb, False)
        right = panel(fB[jb], rb, "B3: certainty-equivalent", "planning μ (estimate)", rb.mu_hat,
                      ended=t > t_end_B)
        W, H = left.width + right.width, left.height
        canvas = Image.new("RGB", (W, H + 40), (20, 20, 20))
        canvas.paste(left, (0, 0)); canvas.paste(right, (left.width, 0))
        d = ImageDraw.Draw(canvas)
        d.text((14, H + 8), f"CARLA Town04 highway, scenario S1 (hidden wet patch)   t = {t:5.1f} s",
               font=F_TXT, fill=(220, 220, 220))
        canvas.save(f"{TMP}/f_{n:05d}.jpg", quality=88)
        if n % 200 == 0:
            print(f"frame {n}/{len(fP)}")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "25",
                    "-i", f"{TMP}/f_%05d.jpg", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "scale=1920:-2", "media/side_by_side_S1.mp4"], check=True)
    print("saved media/side_by_side_S1.mp4")


if __name__ == "__main__":
    main()

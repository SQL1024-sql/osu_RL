"""Makes an osu! skin that draws gameplay the way render.py draws the vision student's training frames, so lazer's
screen looks like what the student learnt from: grey filled circles with a white rim and a dot for the number, a grey
approach circle, a dark slider body with a light border, a small dot for the cursor, and nothing else on the playfield
(no numbers, judgements, hit lighting, follow points, reverse arrows, slider ticks, follow circle, cursor trail,
health bar). Brightnesses are render.py's. Menus, sounds and the results screen stay those of the skin it starts from.

    python -m live.student_skin "C:/Users/User/AppData/Roaming/osu/exports/My skin.osk"   # -> runs/skin/<name> (osu_rl student).osk

Open the .osk with lazer to import it, then pick it in the skin settings. Also turn off "Lighten during breaks".
"""
import argparse
import io
import os
import re
import zipfile

import cv2
import numpy as np

from vision.render import APPROACH, APPROACH_W, BODY, BORDER, FILL, NUMBER, RIM, RIM_W

R = 128  # @2x pixels of a hit circle's radius (legacy skins: 128 px = the diameter at @1x)
SS = 4  # supersampling for smooth edges
CURSOR_R = 8.8 * 2  # @2x px: the student's cursor dot, 2.2 frame px = 8.8 osu! px
# elements replaced by nothing: every variant (@2x, animation frames -N / N) of these names is dropped
BLANK = ["default-" + str(d) for d in range(10)] + [
    "hit0", "hit50", "hit100", "hit100k", "hit300", "hit300g", "hit300k", "lighting", "particle50", "particle100",
    "particle300", "followpoint", "reversearrow", "sliderscorepoint", "sliderfollowcircle", "sliderendcircle",
    "sliderendcircleoverlay", "sliderb-nd", "sliderb-spec", "cursormiddle", "cursortrail",
    "scorebar-bg", "scorebar-colour", "scorebar-ki", "scorebar-kidanger", "scorebar-kidanger2", "scorebar-marker",
    "comboburst", "count1", "count2", "count3", "go", "ready", "play-warningarrow", "section-pass", "section-fail",
]
DRAWN = ["hitcircle", "hitcircleoverlay", "approachcircle", "sliderb", "sliderstartcircle", "cursor"]


def canvas(radius):
    side = int(np.ceil(2 * radius + 8))
    return np.zeros((side * SS, side * SS), np.float32), np.zeros((side * SS, side * SS), np.float32), side


def disk(layer, mask, side, r, value):
    c = side * SS // 2
    cv2.circle(layer, (c, c), int(r * SS), value, -1, cv2.LINE_AA)
    cv2.circle(mask, (c, c), int(r * SS), 1.0, -1, cv2.LINE_AA)


def ring(layer, mask, side, r, width, value):
    c = side * SS // 2
    tmp = np.zeros_like(mask)
    cv2.circle(tmp, (c, c), int(r * SS), 1.0, max(1, int(width * SS)), cv2.LINE_AA)
    layer[:] = layer * (1 - tmp) + value * tmp
    mask[:] = np.maximum(mask, tmp)


def png(layer, mask, side):
    """Grey layer + coverage -> RGBA png bytes; colour is the grey value, alpha the coverage."""
    g = cv2.resize(layer, (side, side), interpolation=cv2.INTER_AREA)
    a = cv2.resize(mask, (side, side), interpolation=cv2.INTER_AREA)
    grey = np.clip(g / np.maximum(a, 1e-6), 0, 1)  # un-premultiply: colour where covered
    img = np.dstack([grey, grey, grey, a])
    return cv2.imencode(".png", (np.clip(img, 0, 1) * 255).round().astype(np.uint8))[1].tobytes()


def blank_png():
    return cv2.imencode(".png", np.zeros((1, 1, 4), np.uint8))[1].tobytes()


def elements():
    """name -> png bytes, all @2x."""
    out = {}
    outer = R * (1 + RIM_W / 2)
    layer, mask, side = canvas(outer)
    disk(layer, mask, side, R, FILL)
    out["hitcircle"] = png(layer, mask, side)
    layer, mask, side = canvas(outer)
    disk(layer, mask, side, R / 4, NUMBER)  # stands in for the combo number, as in render.py
    ring(layer, mask, side, R, RIM_W * R, RIM)
    out["hitcircleoverlay"] = png(layer, mask, side)
    # the approach ring grows to 4x with its texture, so it is drawn thinner than render.py's (constant) width
    layer, mask, side = canvas(R * (1 + APPROACH_W))
    ring(layer, mask, side, R, APPROACH_W * R * 0.6, APPROACH)
    out["approachcircle"] = png(layer, mask, side)
    layer, mask, side = canvas(outer)  # the slider ball, drawn like a hit circle in render.py
    disk(layer, mask, side, R, FILL)
    disk(layer, mask, side, R / 4, NUMBER)
    ring(layer, mask, side, R, RIM_W * R, RIM)
    out["sliderb"] = png(layer, mask, side)
    layer, mask, side = canvas(CURSOR_R + 2)
    disk(layer, mask, side, CURSOR_R, 1.0)
    out["cursor"] = png(layer, mask, side)
    return out


def variants(name, names):
    """Files of the skin that are this element: name.png, name@2x.png, name-0.png, name0@2x.png, ..."""
    pat = re.compile(re.escape(name) + r"(-?\d+)?(@2x)?\.png$", re.I)
    return [n for n in names if pat.fullmatch(os.path.basename(n))]


def skin_ini(text, name):
    def setv(section, key, value, text):
        sec = re.search(rf"^\[{section}\][^\[]*", text, re.M | re.I)
        if sec is None:
            return text + f"\n[{section}]\n{key}: {value}\n"
        body = sec.group(0)
        line = re.compile(rf"^(\s*){key}\s*:.*$", re.M | re.I)
        new = line.sub(rf"\g<1>{key}: {value}", body) if line.search(body) else body.rstrip("\n") + f"\n    {key}: {value}\n\n"
        return text[:sec.start()] + new + text[sec.end():]

    for section, key, value in [
        ("General", "Name", name), ("General", "CursorCentre", 1), ("General", "CursorExpand", 0),
        ("General", "CursorRotate", 0), ("General", "AllowSliderBallTint", 0), ("General", "HitCircleOverlayAboveNumber", 1),
        ("Colours", "SliderBorder", ",".join([str(round(BORDER * 255))] * 3)),
        ("Colours", "SliderTrackOverride", ",".join([str(round(BODY * 255))] * 3)),
    ] + [("Colours", f"Combo{i}", "255,255,255") for i in range(1, 5)]:
        text = setv(section, key, value, text)
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("osk", help="the skin to start from (an exported .osk)")
    p.add_argument("--out", help="default: runs/skin/<name> (osu_rl student).osk")
    args = p.parse_args()
    src = zipfile.ZipFile(args.osk)
    names = [n for n in src.namelist() if not n.endswith("/") and not n.startswith(".")]  # no stray dot-folders
    ini_name = next((n for n in names if n.lower() == "skin.ini"), None)
    ini = src.read(ini_name).decode("utf-8-sig", "replace") if ini_name else "[General]\n"
    old = re.search(r"^\s*Name\s*:\s*(.+)$", ini, re.M | re.I)
    name = f"{old.group(1).strip() if old else 'skin'} (osu_rl student)"
    out = args.out or os.path.join("runs", "skin", re.sub(r'[\\/:*?"<>|]', "_", name) + ".osk")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    drop = {n for e in BLANK + DRAWN for n in variants(e, names)}
    new = elements()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            if n in drop or n == ini_name:
                continue
            z.writestr(n, src.read(n))
        z.writestr("skin.ini", skin_ini(ini, name))
        for e in BLANK:
            z.writestr(f"{e}.png", blank_png())
        for e, data in new.items():
            z.writestr(f"{e}@2x.png", data)
    open(out, "wb").write(buf.getvalue())
    print(f"{name}: kept {len(names) - len(drop) - (ini_name is not None)} files, replaced {len(drop)}, "
          f"blanked {len(BLANK)}, drew {len(new)} -> {out}")


if __name__ == "__main__":
    main()

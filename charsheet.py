# Copyright 2026 YUZUKI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Eight-view character sheets: one Edit per view from a white-background, full-body front reference.

Direction keys and labels follow the common character-sheet convention (and the Qwen-Image-Edit
multiple-angles sheets): `left` is the character turned to their own left, so the nose points to the
right edge of the image; `right` points to the left edge. The 45-degree views sit between those.
"""

import io
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PREP_PROMPT = (
    "Show the same character from the first image as a full-body front view from head to toe, standing upright in a "
    "relaxed neutral pose facing the camera, on a plain white studio background. Keep the face, hair, outfit, "
    "accessories and colors exactly the same."
)

BASE_PROMPT = (
    "Show the same character from the first image, full body from head to toe, standing in a relaxed neutral pose, "
    "on a plain white studio background. Keep the face, hair, outfit, accessories and colors exactly the same. "
)

# Where the character's own left side appears; keeps one-sided details (a single sock, an armband) on their side.
_LEFT_ON_IMAGE_RIGHT = "The character's own left side is on the right of the image, as in the first image."
_LEFT_ON_IMAGE_LEFT = ("Seen from behind, the character's own left side is now on the left of the image: one-sided "
                       "details stay on the same body side as in the first image.")
# Rear views: only the back of the outfit is visible.
_BACK_ONLY = ("Show the back of the outfit as it looks from behind; front-only details such as the face, chest "
              "prints, front pockets, bib, buttons and zippers are not visible.")

VIEWS = [
    {"key": "front", "label_ja": "前", "label_en": "Front",
     "prompt": "A straight front view: the character faces the camera directly, both eyes and the whole face "
               "are visible. " + _LEFT_ON_IMAGE_RIGHT},
    {"key": "back", "label_ja": "後ろ", "label_en": "Back",
     "prompt": "The character has turned around completely: a rear view showing the back of the head, the back of "
               "the outfit and the back of the legs; the face is not visible. " + _BACK_ONLY + " "
               + _LEFT_ON_IMAGE_LEFT},
    {"key": "left", "label_ja": "左", "label_en": "Left",
     "prompt": "The character has turned 90 degrees to their own left: a true side profile, the nose points to the "
               "right edge of the image and only one eye is visible; we see the right side of the body."},
    {"key": "right", "label_ja": "右", "label_en": "Right",
     "prompt": "The character has turned 90 degrees to their own right: a true side profile, the nose points to the "
               "left edge of the image and only one eye is visible; we see the left side of the body."},
    {"key": "front_left_45", "label_ja": "左前45度", "label_en": "Front-Left 45°",
     "prompt": "The character has turned 45 degrees to their own left, exactly halfway between the front view and "
               "the side profile: the face and body are turned toward the right side of the image, the shoulders and "
               "hips are at a clear diagonal, the near shoulder is in front and the far arm is partly hidden behind "
               "the body. " + _LEFT_ON_IMAGE_RIGHT},
    {"key": "front_right_45", "label_ja": "右前45度", "label_en": "Front-Right 45°",
     "prompt": "The character has turned 45 degrees to their own right, exactly halfway between the front view and "
               "the side profile: the face and body are turned toward the left side of the image, the shoulders and "
               "hips are at a clear diagonal, the near shoulder is in front and the far arm is partly hidden behind "
               "the body. " + _LEFT_ON_IMAGE_RIGHT},
    {"key": "back_left_45", "label_ja": "左後ろ45度", "label_en": "Back-Left 45°",
     "prompt": "The character has turned 135 degrees to their own left: a three-quarter rear view, we mostly see "
               "the back and the back of the head, which is turned toward the right side of the image; only the "
               "edge of the cheek shows on the right, the face is not visible. " + _BACK_ONLY + " "
               + _LEFT_ON_IMAGE_LEFT},
    {"key": "back_right_45", "label_ja": "右後ろ45度", "label_en": "Back-Right 45°",
     "prompt": "The character has turned 135 degrees to their own right: a three-quarter rear view, we mostly see "
               "the back and the back of the head, which is turned toward the left side of the image; only the "
               "edge of the cheek shows on the left, the face is not visible. " + _BACK_ONLY + " "
               + _LEFT_ON_IMAGE_LEFT},
]
VIEW_BY_KEY = {view["key"]: view for view in VIEWS}

# Sheet order (same as the existing character sheets): front-facing views on top, rear-facing below.
SHEET_LAYOUT = [
    ["front", "front_left_45", "front_right_45", "left"],
    ["right", "back_left_45", "back_right_45", "back"],
]

PREP_MODES = ("auto", "on", "off")
# prep=auto skips the conversion when at least this share of the image border is white.
WHITE_BACKGROUND_THRESHOLD = 0.85


def view_prompt(key: str) -> str:
    return BASE_PROMPT + VIEW_BY_KEY[key]["prompt"]


def parse_views(raw: str | None) -> list[str]:
    """Comma-separated keys in request order (duplicates dropped); empty means all eight."""
    if not raw or not raw.strip():
        return [view["key"] for view in VIEWS]
    keys = list(dict.fromkeys(key.strip() for key in raw.split(",") if key.strip()))
    invalid = [key for key in keys if key not in VIEW_BY_KEY]
    if invalid or not keys:
        raise ValueError(f"viewsに不正な方向があります: {', '.join(invalid) or raw}。"
                         f"使える方向: {', '.join(VIEW_BY_KEY)}")
    return keys


def white_background_ratio(image: Image.Image) -> float:
    """Share of near-white, unsaturated pixels in a border band (4% of the short side, at least 4 px)."""
    rgb = image.convert("RGB")
    if max(rgb.size) > 512:
        rgb = rgb.copy()
        rgb.thumbnail((512, 512))
    width, height = rgb.size
    band = max(4, round(min(width, height) * 0.04))
    pixels = rgb.load()
    white = total = 0
    for y in range(height):
        edge_row = y < band or y >= height - band
        for x in (range(width) if edge_row else [*range(band), *range(width - band, width)]):
            r, g, b = pixels[x, y]
            total += 1
            if min(r, g, b) >= 225 and max(r, g, b) - min(r, g, b) <= 24:
                white += 1
    return white / max(total, 1)


def has_white_background(image: Image.Image, threshold: float = WHITE_BACKGROUND_THRESHOLD) -> bool:
    return white_background_ratio(image) >= threshold


FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def build_sheet(images: dict[str, Path], cell=(400, 500), image_height=440, margin=20,
                title="Character Sheet") -> Image.Image:
    """4 x 2 grid in SHEET_LAYOUT order with ja/en labels; missing views leave an empty cell."""
    cell_w, cell_h = cell
    cols, rows = len(SHEET_LAYOUT[0]), len(SHEET_LAYOUT)
    title_h = 70
    sheet = Image.new("RGB", (margin * (cols + 1) + cell_w * cols, title_h + margin * (rows + 1) + cell_h * rows),
                      "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 20), title, fill=(30, 30, 30), font=_font(32))
    label_font = _font(18)
    for row, keys in enumerate(SHEET_LAYOUT):
        for col, key in enumerate(keys):
            x = margin + col * (cell_w + margin)
            y = title_h + margin + row * (cell_h + margin)
            draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(220, 220, 220))
            path = images.get(key)
            if path and Path(path).is_file():
                with Image.open(path) as im:
                    fitted = im.convert("RGB")
                fitted.thumbnail((cell_w - 20, image_height - 10), Image.Resampling.LANCZOS)
                sheet.paste(fitted, (x + (cell_w - fitted.width) // 2,
                                     y + 10 + (image_height - 10 - fitted.height) // 2))
            view = VIEW_BY_KEY[key]
            label = f"{view['label_ja']} / {view['label_en']}"
            left, _, right, _ = draw.textbbox((0, 0), label, font=label_font)
            draw.text((x + (cell_w - (right - left)) // 2, y + image_height + 12), label, fill=(30, 30, 30),
                      font=label_font)
    return sheet


def build_zip(files: dict[str, Path]) -> bytes:
    """{name in the archive: path}; missing files are skipped."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            if path and Path(path).is_file():
                archive.write(path, arcname=name)
    return buffer.getvalue()

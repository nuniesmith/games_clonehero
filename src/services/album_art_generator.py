"""
Clone Hero Content Manager - Album Art Generator Service

Generates procedural album art (512x512 PNG) for generated songs using
Pillow.  The art is derived from audio features (tempo, energy, spectral
characteristics) and song metadata so that each song gets a unique but
aesthetically coherent cover.

Clone Hero looks for ``album.png`` (or ``album.jpg``) in each song folder.
This module creates that file during the chart-generation pipeline.

Design approach:
    1. A colour palette is derived from tempo, energy, and genre.
    2. A background gradient is drawn.
    3. Geometric shapes (circles, rectangles, lines, polygons) are placed
       procedurally based on onset strengths and beat patterns.
    4. A stylised waveform or frequency-bar visualisation is overlaid.
    5. Song title and artist text are rendered on top.

No external API or network access is required — everything is generated
locally with Pillow.
"""

from __future__ import annotations

import colorsys
import hashlib
import math
import random
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
except ImportError:
    Image = None  # type: ignore[assignment,misc]
    ImageDraw = None  # type: ignore[assignment,misc]
    ImageFilter = None  # type: ignore[assignment,misc]
    ImageFont = None  # type: ignore[assignment,misc]
    logger.warning(
        "⚠️ Pillow is not installed — album art generation will be disabled. "
        "Install it with: pip install Pillow"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ART_SIZE = 512  # Output image dimensions (square)
MARGIN = 32  # Text margin from edges
MIN_FONT_SIZE = 16
MAX_FONT_TITLE = 42
MAX_FONT_ARTIST = 28

# ---------------------------------------------------------------------------
# Colour palette generation
# ---------------------------------------------------------------------------

# Genre-to-hue base mapping (hue 0.0–1.0)
GENRE_HUE_MAP: Dict[str, float] = {
    "rock": 0.02,  # red-orange
    "metal": 0.75,  # deep purple
    "punk": 0.95,  # hot pink / magenta
    "pop": 0.55,  # cyan-blue
    "dance": 0.83,  # violet
    "electronic": 0.48,  # teal
    "edm": 0.50,  # teal-cyan
    "techno": 0.60,  # blue
    "house": 0.45,  # teal
    "trance": 0.70,  # indigo
    "ambient": 0.35,  # green
    "chill": 0.40,  # green-teal
    "lofi": 0.10,  # warm orange
    "jazz": 0.12,  # amber
    "classical": 0.58,  # sky blue
    "folk": 0.15,  # golden
    "acoustic": 0.13,  # warm gold
    "hip-hop": 0.08,  # orange-red
    "rap": 0.00,  # red
    "r&b": 0.90,  # magenta-pink
    "blues": 0.62,  # blue
    "country": 0.10,  # warm amber
    "reggae": 0.30,  # yellow-green
    "funk": 0.05,  # red-orange
    "soul": 0.88,  # pink
    "default": 0.55,  # blue
    "generated": 0.55,  # blue
}


def _genre_base_hue(genre: str) -> float:
    """Return the base hue (0.0–1.0) for a genre string."""
    genre_lower = (genre or "").lower().strip()
    for keyword, hue in GENRE_HUE_MAP.items():
        if keyword in genre_lower:
            return hue
    return GENRE_HUE_MAP["default"]


def _name_hash(name: str) -> float:
    """Deterministic float 0.0–1.0 from a string (for variation)."""
    digest = hashlib.md5(name.encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    """Convert HSL (all 0.0–1.0) to an (R, G, B) tuple (0–255)."""
    r, g, b = colorsys.hls_to_rgb(h % 1.0, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


def generate_palette(
    tempo: float,
    energy: float,
    genre: str,
    song_name: str,
    artist: str,
    rng: random.Random,
) -> Dict[str, Tuple[int, int, int]]:
    """
    Generate a cohesive colour palette for the album art.

    Returns a dict with keys:
        bg_top, bg_bottom   – gradient background colours
        accent1, accent2    – primary accent colours (shapes / highlights)
        shape_fill          – semi-transparent shape fill base
        text_primary        – title text colour
        text_secondary      – artist text colour
        glow                – glow / overlay colour
    """
    base_hue = _genre_base_hue(genre)

    # Add variation from the song name so every song is different
    variation = _name_hash(song_name + artist)
    hue = (base_hue + variation * 0.15 - 0.075) % 1.0

    # Tempo affects saturation (faster = more vivid)
    tempo_norm = max(0.0, min(1.0, (tempo - 60) / 140))  # 60–200 BPM range
    saturation = 0.45 + tempo_norm * 0.40  # 0.45–0.85

    # Energy affects lightness
    energy_norm = max(0.0, min(1.0, energy * 5))  # rough normalisation
    lightness_bg = 0.10 + energy_norm * 0.10  # dark background: 0.10–0.20

    # Complementary / analogous hue for accent
    accent_hue = (hue + 0.35 + rng.uniform(-0.05, 0.05)) % 1.0
    accent_hue2 = (hue + 0.55 + rng.uniform(-0.05, 0.05)) % 1.0

    return {
        "bg_top": _hsl_to_rgb(hue, saturation * 0.7, lightness_bg),
        "bg_bottom": _hsl_to_rgb(
            (hue + 0.08) % 1.0, saturation * 0.6, lightness_bg * 0.7
        ),
        "accent1": _hsl_to_rgb(accent_hue, saturation, 0.55),
        "accent2": _hsl_to_rgb(accent_hue2, saturation * 0.8, 0.45),
        "shape_fill": _hsl_to_rgb(hue, saturation * 0.5, 0.30),
        "text_primary": (255, 255, 255),
        "text_secondary": _hsl_to_rgb(hue, 0.20, 0.80),
        "glow": _hsl_to_rgb(hue, saturation, 0.50),
    }


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _draw_gradient(
    draw: "ImageDraw.ImageDraw",
    size: int,
    top_color: Tuple[int, int, int],
    bottom_color: Tuple[int, int, int],
) -> None:
    """Draw a vertical linear gradient background."""
    for y in range(size):
        t = y / max(1, size - 1)
        r = int(top_color[0] + (bottom_color[0] - top_color[0]) * t)
        g = int(top_color[1] + (bottom_color[1] - top_color[1]) * t)
        b = int(top_color[2] + (bottom_color[2] - top_color[2]) * t)
        draw.line([(0, y), (size - 1, y)], fill=(r, g, b))


def _draw_radial_glow(
    img: "Image.Image",
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    intensity: int = 60,
) -> None:
    """Draw a soft radial glow at the given centre."""
    glow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)

    steps = max(6, radius // 4)
    for i in range(steps):
        t = i / max(1, steps - 1)
        r = int(radius * (1.0 - t * 0.7))
        alpha = int(intensity * (1.0 - t))
        if r < 1 or alpha < 1:
            continue
        x0 = center[0] - r
        y0 = center[1] - r
        x1 = center[0] + r
        y1 = center[1] + r
        glow_draw.ellipse(
            [x0, y0, x1, y1],
            fill=(color[0], color[1], color[2], alpha),
        )

    img.paste(Image.alpha_composite(img.convert("RGBA"), glow_layer).convert("RGB"))


def _draw_geometric_shapes(
    draw: "ImageDraw.ImageDraw",
    img: "Image.Image",
    size: int,
    palette: Dict[str, Tuple[int, int, int]],
    onset_strengths: List[float],
    beat_times: List[float],
    tempo: float,
    rng: random.Random,
) -> None:
    """Draw procedural geometric shapes based on audio features."""
    num_shapes = max(5, min(30, len(onset_strengths) // 8))

    for i in range(num_shapes):
        strength = (
            onset_strengths[i * len(onset_strengths) // num_shapes]
            if onset_strengths
            else 0.5
        )
        shape_type = rng.choice(["circle", "rect", "line", "triangle", "diamond"])

        # Position influenced by beat pattern
        cx = rng.randint(MARGIN, size - MARGIN)
        cy = rng.randint(MARGIN, size - MARGIN)
        shape_size = int(20 + strength * 80 + rng.uniform(0, 30))

        # Colour: accent colours with varying alpha
        base_color = rng.choice(
            [palette["accent1"], palette["accent2"], palette["shape_fill"]]
        )
        alpha = int(40 + strength * 100)
        alpha = min(180, alpha)
        fill_rgba = (base_color[0], base_color[1], base_color[2], alpha)

        # Use an overlay for transparency
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        if shape_type == "circle":
            r = shape_size // 2
            overlay_draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=fill_rgba,
                outline=None,
            )
        elif shape_type == "rect":
            w = shape_size
            h = int(shape_size * rng.uniform(0.5, 1.5))
            angle = rng.uniform(0, 45) if rng.random() < 0.5 else 0
            if angle == 0:
                overlay_draw.rectangle(
                    [cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2],
                    fill=fill_rgba,
                )
            else:
                # Rotated rectangle approximated as a polygon
                corners = [
                    (-w // 2, -h // 2),
                    (w // 2, -h // 2),
                    (w // 2, h // 2),
                    (-w // 2, h // 2),
                ]
                rad = math.radians(angle)
                rotated = [
                    (
                        int(cx + x * math.cos(rad) - y * math.sin(rad)),
                        int(cy + x * math.sin(rad) + y * math.cos(rad)),
                    )
                    for x, y in corners
                ]
                overlay_draw.polygon(rotated, fill=fill_rgba)
        elif shape_type == "line":
            length = shape_size * 2
            angle = rng.uniform(0, math.pi * 2)
            x2 = int(cx + length * math.cos(angle))
            y2 = int(cy + length * math.sin(angle))
            line_width = max(1, int(2 + strength * 4))
            line_color = (base_color[0], base_color[1], base_color[2], alpha + 30)
            overlay_draw.line(
                [(cx, cy), (x2, y2)],
                fill=line_color,
                width=line_width,
            )
        elif shape_type == "triangle":
            r = shape_size // 2
            angle_offset = rng.uniform(0, math.pi * 2)
            points = []
            for j in range(3):
                a = angle_offset + j * (2 * math.pi / 3)
                px = int(cx + r * math.cos(a))
                py = int(cy + r * math.sin(a))
                points.append((px, py))
            overlay_draw.polygon(points, fill=fill_rgba)
        elif shape_type == "diamond":
            r = shape_size // 2
            points = [
                (cx, cy - r),
                (cx + r, cy),
                (cx, cy + r),
                (cx - r, cy),
            ]
            overlay_draw.polygon(points, fill=fill_rgba)

        # Composite the overlay
        composite = Image.alpha_composite(img.convert("RGBA"), overlay)
        img.paste(composite.convert("RGB"))


def _draw_waveform_bars(
    draw: "ImageDraw.ImageDraw",
    size: int,
    palette: Dict[str, Tuple[int, int, int]],
    onset_strengths: List[float],
    rng: random.Random,
) -> None:
    """Draw a stylised frequency-bar visualisation across the middle."""
    if not onset_strengths:
        return

    num_bars = min(64, len(onset_strengths))
    bar_width = max(2, (size - MARGIN * 2) // num_bars - 1)
    max_bar_height = size // 4

    # Sample onset strengths evenly
    step = max(1, len(onset_strengths) // num_bars)
    sampled = [
        onset_strengths[i * step]
        for i in range(num_bars)
        if i * step < len(onset_strengths)
    ]
    if not sampled:
        return

    # Normalise
    max_s = max(sampled) if max(sampled) > 0 else 1.0
    normalised = [s / max_s for s in sampled]

    baseline_y = size // 2 + size // 8  # slightly below centre

    for i, strength in enumerate(normalised):
        x = MARGIN + i * (bar_width + 1)
        bar_h = int(max_bar_height * strength * rng.uniform(0.7, 1.0))
        if bar_h < 2:
            bar_h = 2

        # Colour gradient from accent1 to accent2
        t = i / max(1, len(normalised) - 1)
        c1 = palette["accent1"]
        c2 = palette["accent2"]
        r = int(c1[0] + (c2[0] - c1[0]) * t)
        g = int(c1[1] + (c2[1] - c1[1]) * t)
        b = int(c1[2] + (c2[2] - c1[2]) * t)

        # Draw bar going upward
        draw.rectangle(
            [x, baseline_y - bar_h, x + bar_width, baseline_y],
            fill=(r, g, b, 200),
        )
        # Mirror below (smaller)
        mirror_h = bar_h // 3
        draw.rectangle(
            [x, baseline_y + 2, x + bar_width, baseline_y + 2 + mirror_h],
            fill=(r, g, b, 80),
        )


def _draw_concentric_rings(
    draw: "ImageDraw.ImageDraw",
    size: int,
    palette: Dict[str, Tuple[int, int, int]],
    tempo: float,
    rng: random.Random,
) -> None:
    """Draw concentric rings centred on the image — a vinyl/record motif."""
    cx, cy = size // 2, size // 2
    num_rings = max(3, min(12, int(tempo / 20)))
    max_radius = size // 3

    for i in range(num_rings):
        r = int(max_radius * (i + 1) / num_rings)
        t = i / max(1, num_rings - 1)
        c = palette["accent1"] if i % 2 == 0 else palette["accent2"]
        alpha = int(30 + t * 50)
        width = max(1, int(1 + (1.0 - t) * 2))
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=(c[0], c[1], c[2], alpha),
            width=width,
        )


def _draw_grid_dots(
    draw: "ImageDraw.ImageDraw",
    size: int,
    palette: Dict[str, Tuple[int, int, int]],
    rng: random.Random,
) -> None:
    """Draw a subtle dot grid pattern across the background."""
    spacing = rng.choice([24, 32, 40, 48])
    dot_r = max(1, spacing // 12)
    color = palette["shape_fill"]
    alpha = 25

    for x in range(spacing // 2, size, spacing):
        for y in range(spacing // 2, size, spacing):
            jx = x + rng.randint(-2, 2)
            jy = y + rng.randint(-2, 2)
            draw.ellipse(
                [jx - dot_r, jy - dot_r, jx + dot_r, jy + dot_r],
                fill=(color[0], color[1], color[2], alpha),
            )


def _draw_diagonal_stripes(
    draw: "ImageDraw.ImageDraw",
    size: int,
    color: Tuple[int, int, int],
    rng: random.Random,
) -> None:
    """Draw subtle diagonal stripes across the image."""
    stripe_gap = rng.choice([30, 40, 50, 60])
    stripe_width = max(1, stripe_gap // 8)
    alpha = rng.randint(10, 30)
    fill = (color[0], color[1], color[2], alpha)

    for offset in range(-size, size * 2, stripe_gap):
        x0 = offset
        y0 = 0
        x1 = offset - size
        y1 = size
        draw.line([(x0, y0), (x1, y1)], fill=fill, width=stripe_width)


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _get_font(size: int) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    """
    Try to load a TrueType font at the given size.

    Falls back to the Pillow default bitmap font if no TTF is available.
    Tries several common system font paths.
    """
    # Candidate font paths (cross-platform)
    font_candidates = [
        # Linux (Debian/Ubuntu)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        # Windows
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
    ]

    for path in font_candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError, AttributeError):
            continue

    # Last resort: try common font names (some Pillow builds support this)
    for name in ["DejaVuSans-Bold", "DejaVuSans", "Arial", "Helvetica"]:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError, AttributeError):
            continue

    # Final fallback: default bitmap font (no size control)
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def _fit_text_size(
    text: str,
    max_width: int,
    max_font_size: int,
    min_font_size: int = MIN_FONT_SIZE,
) -> int:
    """Find the largest font size that fits the text within max_width."""
    for size in range(max_font_size, min_font_size - 1, -2):
        font = _get_font(size)
        try:
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
        except AttributeError:
            # Older Pillow without getbbox
            try:
                text_width = font.getlength(text)
            except AttributeError:
                text_width = len(text) * size * 0.6
        if text_width <= max_width:
            return size
    return min_font_size


def _draw_text_with_shadow(
    draw: "ImageDraw.ImageDraw",
    position: Tuple[int, int],
    text: str,
    font: "ImageFont.FreeTypeFont | ImageFont.ImageFont",
    fill: Tuple[int, int, int],
    shadow_color: Tuple[int, int, int] = (0, 0, 0),
    shadow_offset: int = 2,
) -> None:
    """Draw text with a drop shadow for readability."""
    x, y = position
    # Shadow
    draw.text(
        (x + shadow_offset, y + shadow_offset),
        text,
        font=font,
        fill=shadow_color,
    )
    # Slightly offset second shadow for more depth
    draw.text(
        (x + shadow_offset // 2, y + shadow_offset // 2),
        text,
        font=font,
        fill=(shadow_color[0], shadow_color[1], shadow_color[2]),
    )
    # Main text
    draw.text((x, y), text, font=font, fill=fill)


def _truncate_text(text: str, max_chars: int = 30) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _render_title_and_artist(
    img: "Image.Image",
    draw: "ImageDraw.ImageDraw",
    size: int,
    title: str,
    artist: str,
    palette: Dict[str, Tuple[int, int, int]],
) -> None:
    """Render the song title and artist name at the bottom of the image."""
    max_text_width = size - MARGIN * 2

    # Truncate if needed
    display_title = _truncate_text(title, 35)
    display_artist = _truncate_text(artist, 40)

    # Fit fonts
    title_size = _fit_text_size(display_title, max_text_width, MAX_FONT_TITLE)
    artist_size = _fit_text_size(display_artist, max_text_width, MAX_FONT_ARTIST)

    title_font = _get_font(title_size)
    artist_font = _get_font(artist_size)

    # Calculate positions (bottom-left, stacked)
    line_gap = 6
    try:
        title_bbox = title_font.getbbox(display_title)
        title_h = title_bbox[3] - title_bbox[1]
    except AttributeError:
        title_h = title_size

    try:
        artist_bbox = artist_font.getbbox(display_artist)
        artist_h = artist_bbox[3] - artist_bbox[1]
    except AttributeError:
        artist_h = artist_size

    total_text_h = title_h + line_gap + artist_h
    text_y_start = size - MARGIN - total_text_h

    # Semi-transparent backdrop behind text for readability
    backdrop_padding = 12
    backdrop = Image.new("RGBA", img.size, (0, 0, 0, 0))
    bd_draw = ImageDraw.Draw(backdrop)
    bd_draw.rectangle(
        [
            0,
            text_y_start - backdrop_padding,
            size,
            size,
        ],
        fill=(0, 0, 0, 120),
    )
    composite = Image.alpha_composite(img.convert("RGBA"), backdrop)
    img.paste(composite.convert("RGB"))

    # Re-create draw object after paste
    draw = ImageDraw.Draw(img)

    # Draw title
    _draw_text_with_shadow(
        draw,
        (MARGIN, text_y_start),
        display_title,
        title_font,
        fill=palette["text_primary"],
    )

    # Draw artist
    _draw_text_with_shadow(
        draw,
        (MARGIN, text_y_start + title_h + line_gap),
        display_artist,
        artist_font,
        fill=palette["text_secondary"],
    )


# ---------------------------------------------------------------------------
# Art style selectors
# ---------------------------------------------------------------------------


def _style_geometric(
    img: "Image.Image",
    draw: "ImageDraw.ImageDraw",
    size: int,
    palette: Dict[str, Tuple[int, int, int]],
    onset_strengths: List[float],
    beat_times: List[float],
    tempo: float,
    rng: random.Random,
) -> None:
    """Art style: abstract geometric shapes."""
    _draw_diagonal_stripes(draw, size, palette["shape_fill"], rng)
    _draw_geometric_shapes(
        draw, img, size, palette, onset_strengths, beat_times, tempo, rng
    )
    _draw_radial_glow(
        img, (size // 2, size // 3), size // 3, palette["glow"], intensity=40
    )


def _style_waveform(
    img: "Image.Image",
    draw: "ImageDraw.ImageDraw",
    size: int,
    palette: Dict[str, Tuple[int, int, int]],
    onset_strengths: List[float],
    beat_times: List[float],
    tempo: float,
    rng: random.Random,
) -> None:
    """Art style: frequency bar visualisation."""
    _draw_grid_dots(draw, size, palette, rng)
    _draw_waveform_bars(draw, size, palette, onset_strengths, rng)
    _draw_radial_glow(
        img, (size // 2, size // 2), size // 4, palette["accent1"], intensity=30
    )


def _style_vinyl(
    img: "Image.Image",
    draw: "ImageDraw.ImageDraw",
    size: int,
    palette: Dict[str, Tuple[int, int, int]],
    onset_strengths: List[float],
    beat_times: List[float],
    tempo: float,
    rng: random.Random,
) -> None:
    """Art style: concentric rings (vinyl record motif)."""
    _draw_concentric_rings(draw, size, palette, tempo, rng)
    _draw_radial_glow(
        img, (size // 2, size // 2), size // 5, palette["glow"], intensity=50
    )
    # Add a few small geometric accents
    for _ in range(rng.randint(3, 8)):
        cx = rng.randint(MARGIN, size - MARGIN)
        cy = rng.randint(MARGIN, size - MARGIN)
        r = rng.randint(3, 12)
        c = rng.choice([palette["accent1"], palette["accent2"]])
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=(c[0], c[1], c[2], rng.randint(40, 100)),
        )


def _style_nebula(
    img: "Image.Image",
    draw: "ImageDraw.ImageDraw",
    size: int,
    palette: Dict[str, Tuple[int, int, int]],
    onset_strengths: List[float],
    beat_times: List[float],
    tempo: float,
    rng: random.Random,
) -> None:
    """Art style: nebula — multiple overlapping radial glows."""
    num_glows = rng.randint(4, 8)
    for _ in range(num_glows):
        cx = rng.randint(size // 6, size * 5 // 6)
        cy = rng.randint(size // 6, size * 5 // 6)
        radius = rng.randint(size // 6, size // 3)
        color = rng.choice([palette["accent1"], palette["accent2"], palette["glow"]])
        intensity = rng.randint(20, 50)
        _draw_radial_glow(img, (cx, cy), radius, color, intensity=intensity)

    # Scatter some small bright dots (stars)
    draw = ImageDraw.Draw(img)
    num_stars = rng.randint(30, 80)
    for _ in range(num_stars):
        sx = rng.randint(0, size - 1)
        sy = rng.randint(0, size - 1)
        sr = rng.randint(1, 3)
        brightness = rng.randint(180, 255)
        draw.ellipse(
            [sx - sr, sy - sr, sx + sr, sy + sr],
            fill=(brightness, brightness, brightness, rng.randint(100, 220)),
        )


def _style_circuit(
    img: "Image.Image",
    draw: "ImageDraw.ImageDraw",
    size: int,
    palette: Dict[str, Tuple[int, int, int]],
    onset_strengths: List[float],
    beat_times: List[float],
    tempo: float,
    rng: random.Random,
) -> None:
    """Art style: circuit-board lines — good for electronic genres."""
    grid_spacing = rng.choice([32, 48, 64])
    line_color = palette["accent1"]
    alpha = 50

    # Horizontal traces
    for y in range(grid_spacing, size, grid_spacing):
        segments = rng.randint(1, 4)
        for _ in range(segments):
            x1 = rng.randint(0, size // 2)
            x2 = x1 + rng.randint(size // 6, size // 2)
            x2 = min(x2, size)
            draw.line(
                [(x1, y), (x2, y)],
                fill=(line_color[0], line_color[1], line_color[2], alpha),
                width=rng.choice([1, 2]),
            )
            # Node dot at junction
            if rng.random() < 0.4:
                nr = rng.randint(2, 5)
                draw.ellipse(
                    [x2 - nr, y - nr, x2 + nr, y + nr],
                    fill=(
                        palette["accent2"][0],
                        palette["accent2"][1],
                        palette["accent2"][2],
                        alpha + 40,
                    ),
                )

    # Vertical traces
    for x in range(grid_spacing, size, grid_spacing):
        segments = rng.randint(1, 3)
        for _ in range(segments):
            y1 = rng.randint(0, size // 2)
            y2 = y1 + rng.randint(size // 6, size // 2)
            y2 = min(y2, size)
            draw.line(
                [(x, y1), (x, y2)],
                fill=(line_color[0], line_color[1], line_color[2], alpha),
                width=rng.choice([1, 2]),
            )

    # Central glow
    _draw_radial_glow(
        img, (size // 2, size // 2), size // 4, palette["glow"], intensity=35
    )

    # Waveform overlay
    _draw_waveform_bars(ImageDraw.Draw(img), size, palette, onset_strengths, rng)


# Art styles registry
ART_STYLES = {
    "geometric": _style_geometric,
    "waveform": _style_waveform,
    "vinyl": _style_vinyl,
    "nebula": _style_nebula,
    "circuit": _style_circuit,
}

# Genre-to-preferred-style hints
GENRE_STYLE_HINTS: Dict[str, List[str]] = {
    "rock": ["geometric", "vinyl"],
    "metal": ["geometric", "nebula"],
    "punk": ["geometric", "waveform"],
    "pop": ["nebula", "waveform"],
    "electronic": ["circuit", "waveform"],
    "edm": ["circuit", "waveform"],
    "techno": ["circuit", "waveform"],
    "ambient": ["nebula", "vinyl"],
    "chill": ["nebula", "vinyl"],
    "jazz": ["vinyl", "nebula"],
    "classical": ["vinyl", "nebula"],
    "default": ["geometric", "waveform", "vinyl", "nebula", "circuit"],
}


def _pick_style(genre: str, rng: random.Random) -> str:
    """Pick an art style based on genre, with some randomness."""
    genre_lower = (genre or "").lower().strip()
    for keyword, styles in GENRE_STYLE_HINTS.items():
        if keyword in genre_lower:
            return rng.choice(styles)
    return rng.choice(GENRE_STYLE_HINTS["default"])


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def generate_album_art(
    output_path: Path,
    song_name: str,
    artist: str,
    tempo: float,
    duration: float,
    onset_strengths: Optional[List[float]] = None,
    beat_times: Optional[List[float]] = None,
    genre: str = "Generated",
    energy: float = 0.5,
    seed: Optional[int] = None,
    size: int = ART_SIZE,
) -> bool:
    """
    Generate a procedural album art PNG and save it to *output_path*.

    Parameters
    ----------
    output_path : Path
        Where to save the PNG file (typically ``album.png`` in the song folder).
    song_name : str
        Song title — used for colour variation and text overlay.
    artist : str
        Artist name — rendered on the cover.
    tempo : float
        Detected tempo in BPM — influences colour saturation and shape density.
    duration : float
        Song duration in seconds.
    onset_strengths : list of float, optional
        Onset strengths from audio analysis — drives visualisation intensity.
    beat_times : list of float, optional
        Beat times from audio analysis.
    genre : str
        Genre string for theme selection.
    energy : float
        Average RMS energy (0.0–1.0 ish) — affects palette brightness.
    seed : int, optional
        Random seed for reproducible output.
    size : int
        Output image size in pixels (square). Default 512.

    Returns
    -------
    bool
        True on success, False on failure (e.g. Pillow not installed).
    """
    if Image is None:
        logger.warning("⚠️ Pillow not installed — skipping album art generation")
        return False

    try:
        rng = random.Random(seed or hash(song_name + artist))

        if onset_strengths is None:
            onset_strengths = [rng.uniform(0.3, 0.9) for _ in range(64)]
        if beat_times is None:
            beat_times = [
                i * (60.0 / max(60, tempo)) for i in range(int(duration * tempo / 60))
            ]

        # Generate colour palette
        palette = generate_palette(tempo, energy, genre, song_name, artist, rng)

        # Create base image (RGBA for compositing)
        img = Image.new("RGBA", (size, size), (0, 0, 0, 255))
        draw = ImageDraw.Draw(img)

        # Step 1: Background gradient
        _draw_gradient(draw, size, palette["bg_top"], palette["bg_bottom"])

        # Step 2: Pick and apply art style
        style_name = _pick_style(genre, rng)
        style_fn = ART_STYLES.get(style_name, _style_geometric)

        logger.debug(
            "🎨 Album art style: {} (genre: {}, tempo: {:.0f})",
            style_name,
            genre,
            tempo,
        )

        style_fn(img, draw, size, palette, onset_strengths, beat_times, tempo, rng)

        # Step 3: Optional blur pass for softer feel on chill/ambient genres
        genre_lower = (genre or "").lower()
        if any(
            kw in genre_lower for kw in ("ambient", "chill", "lofi", "lo-fi", "jazz")
        ):
            try:
                img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
            except Exception:
                pass

        # Step 4: Render title and artist text
        draw = ImageDraw.Draw(img)
        _render_title_and_artist(img, draw, size, song_name, artist, palette)

        # Step 5: Convert to RGB and save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_img = img.convert("RGB")
        final_img.save(str(output_path), "PNG", optimize=True)

        file_size_kb = output_path.stat().st_size / 1024
        logger.info(
            "🎨 Generated album art: {} ({:.0f} KB, style: {})",
            output_path,
            file_size_kb,
            style_name,
        )
        return True

    except Exception as e:
        logger.error("❌ Failed to generate album art: {}", e)
        return False


def generate_album_art_bytes(
    song_name: str,
    artist: str,
    tempo: float,
    duration: float,
    onset_strengths: Optional[List[float]] = None,
    beat_times: Optional[List[float]] = None,
    genre: str = "Generated",
    energy: float = 0.5,
    seed: Optional[int] = None,
    size: int = ART_SIZE,
) -> Optional[bytes]:
    """
    Generate album art and return it as PNG bytes (for API responses).

    Same as :func:`generate_album_art` but returns bytes instead of
    writing to a file.  Returns ``None`` on failure.
    """
    if Image is None:
        return None

    try:
        # Use a temporary path internally
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        ok = generate_album_art(
            output_path=tmp_path,
            song_name=song_name,
            artist=artist,
            tempo=tempo,
            duration=duration,
            onset_strengths=onset_strengths,
            beat_times=beat_times,
            genre=genre,
            energy=energy,
            seed=seed,
            size=size,
        )

        if ok and tmp_path.exists():
            data = tmp_path.read_bytes()
            tmp_path.unlink(missing_ok=True)
            return data

        tmp_path.unlink(missing_ok=True)
        return None

    except Exception as e:
        logger.error("❌ Failed to generate album art bytes: {}", e)
        return None


def is_available() -> bool:
    """Check whether album art generation is available (Pillow installed)."""
    return Image is not None

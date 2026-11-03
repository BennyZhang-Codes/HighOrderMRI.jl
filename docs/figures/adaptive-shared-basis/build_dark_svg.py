#!/usr/bin/env python3
"""Generate the docs dark SVG from the unchanged author export.

Run: python docs/figures/adaptive-shared-basis/build_dark_svg.py
Only colors and the standalone canvas background change; paths and glyphs
remain unchanged. Unknown colors fail so a new export can be reviewed.
"""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

PALETTE = {
    "#000000": "#e6e6eb", "#202020": "#e6e6eb",
    "#545454": "#b9bbc6", "#555555": "#b9bbc6",
    "#355fa0": "#8fb8ff", "#167d78": "#69d0bd",
    "#8a2436": "#f19aab", "#a45e20": "#e9b577",
    "#ffffff": "#23242b", "#f7f7f7": "#24252c",
    "#eaf5f2": "#193d38", "#deedec": "#193d38",
    "#f7f1eb": "#3a2c22", "#909090": "#777b89",
    "#bcbcbc": "#4e515e", "#d7d7d7": "#4e515e",
    "#509d9a": "#5fa89e", "#a2cbc9": "#3d7970",
    "#b9d8d6": "#3d7970",
}


def recolor(match):
    property_name, value = match.groups()
    if value == "none":
        return match.group(0)
    parts = re.fullmatch(r"rgb\(([\d.]+)%,([\d.]+)%,([\d.]+)%\)", value)
    if not parts:
        raise ValueError(f"Unrecognized SVG color: {value}")
    color = "#" + "".join(f"{round(float(part) * 255 / 100):02x}"
                          for part in parts.groups())
    if color not in PALETTE:
        raise ValueError(f"Add an explicit dark palette mapping for {color}")
    return f"{property_name}:{PALETTE[color]};"


def main():
    docs = Path(__file__).resolve().parents[2]
    public = docs / "src/public/figures/adaptive-shared-basis"
    source = public / "Fig1_adaptive_shared_basis.svg"
    target = public / "Fig1_adaptive_shared_basis_dark.svg"
    original = source.read_text()
    root = ET.fromstring(original)
    x, y, width, height = root.attrib["viewBox"].split()
    themed = re.sub(r"(fill|stroke):([^;]+);", recolor, original)
    # A canvas background also makes the SVG readable when opened on its own.
    background = (f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
                  'fill="#1b1b1f"/>')
    themed = re.sub(r"(<svg\b[^>]*)(>)", r'\1 style="fill:#e6e6eb;"\2', themed, count=1)
    themed = re.sub(r"(<svg\b[^>]*>)", lambda m: m.group(0) + "\n" + background,
                    themed, count=1)
    ET.fromstring(themed)
    target.write_text(themed)
    print(target)


if __name__ == "__main__":
    main()

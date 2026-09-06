"""
Validation outputs: see what the pipeline actually measured.

Two artifacts, because they serve different checks:

* **finder chart (PNG)** — a stretched image of the reference frame with a
  green circle on the target and red circles on each comparison star,
  labeled and annotated. Open it anywhere, glance at it, spot a mistake.

* **region file (.reg)** — DS9 region format, which both DS9 and
  AstroImageJ load as an overlay on the original FITS. This is the one to
  use when you want to blink between the overlay and the pixels at full
  resolution, or verify against AIJ's own aperture placement.

FITS itself stores pixels and headers, not colored graphics — the region
file is how that overlay is conventionally carried alongside it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_region_file(path: Path, target_xy: tuple[float, float],
                      comp_xy: list[tuple[float, float]],
                      r_ap: float, r_in: float, r_out: float,
                      comp_labels: list[str] | None = None) -> Path:
    """
    DS9-format region file in image (pixel) coordinates.

    Loads in DS9 (Region > Load Regions) and in AstroImageJ
    (File > Import > Region file), overlaid on the same FITS frame.
    Apertures and sky annuli are both drawn, so you can confirm the
    background annulus isn't sitting on a neighboring star.
    """
    tx, ty = target_xy
    lines = [
        "# Region file format: DS9 version 4.1",
        "# transitphot aperture placement — green: target, red: comparisons",
        'global dashlist=8 3 width=1 font="helvetica 10 normal roman" '
        "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        "image",
        # target: aperture + annulus
        f"circle({tx + 1:.2f},{ty + 1:.2f},{r_ap:.2f}) # color=green width=2 "
        f'text={{TARGET}}',
        f"annulus({tx + 1:.2f},{ty + 1:.2f},{r_in:.2f},{r_out:.2f}) "
        "# color=green dash=1",
    ]
    for i, (cx, cy) in enumerate(comp_xy):
        label = comp_labels[i] if comp_labels else f"C{i + 1}"
        lines.append(
            f"circle({cx + 1:.2f},{cy + 1:.2f},{r_ap:.2f}) # color=red width=2 "
            f"text={{{label}}}"
        )
        lines.append(
            f"annulus({cx + 1:.2f},{cy + 1:.2f},{r_in:.2f},{r_out:.2f}) "
            "# color=red dash=1"
        )
    # DS9/AIJ region files are 1-indexed; pixel arrays are 0-indexed, hence +1
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def make_finder_chart(data: np.ndarray, target_xy: tuple[float, float],
                      comp_xy: list[tuple[float, float]],
                      r_ap: float, r_in: float, r_out: float,
                      out: Path = Path("finder.png"),
                      comp_labels: list[str] | None = None,
                      title: str = "") -> Path:
    """Annotated image of the reference frame — the quick visual check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    # asinh stretch between robust percentiles: shows faint comparisons
    # without blowing out the bright target
    lo, hi = np.nanpercentile(data, [10, 99.5])
    scaled = np.arcsinh(np.clip((data - lo) / max(hi - lo, 1e-9), 0, None))

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(scaled, origin="lower", cmap="gray",
              vmin=0, vmax=np.nanpercentile(scaled, 99.7))

    def mark(xy, color, label):
        x, y = xy
        ax.add_patch(Circle((x, y), r_ap, fill=False, color=color, lw=1.8))
        ax.add_patch(Circle((x, y), r_in, fill=False, color=color, lw=0.7,
                            ls=":", alpha=0.7))
        ax.add_patch(Circle((x, y), r_out, fill=False, color=color, lw=0.7,
                            ls=":", alpha=0.7))
        ax.text(x + r_out + 4, y, label, color=color, fontsize=10,
                va="center", fontweight="bold")

    mark(target_xy, "#2ecc71", "TARGET")
    for i, xy in enumerate(comp_xy):
        mark(xy, "#e74c3c", comp_labels[i] if comp_labels else f"C{i + 1}")

    ax.set_title(title or "Aperture placement — verify before trusting the curve",
                 fontsize=11)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return Path(out)


def save_reference_frame(data: np.ndarray, header, out: Path) -> Path:
    """Write the aligned reference frame the regions refer to, so the
    overlay and the pixels always match."""
    from astropy.io import fits

    fits.PrimaryHDU(data=data.astype("float32"), header=header).writeto(
        out, overwrite=True)
    return Path(out)

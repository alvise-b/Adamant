"""
Quaternary cemented carbide visualizer — TiC / NbC / Ni / Fe tetrahedron.

Each corner of the tetrahedron is a pure end-member.
Point position  → composition (wt%).
Point color     → Vickers hardness HV  (gray at 100 HV → red at max HV).

Usage
-----
    python tetrahedron_plot.py                          # uses materials_dataset.json
    python tetrahedron_plot.py path/to/other.json       # alternative dataset

Output: tetrahedron_plot.html  (standalone, open in any browser)
"""

import json
import sys
import os
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Tetrahedron geometry  (regular tetrahedron, edge length = 1)
# Corners: A=TiC  B=NbC  C=Ni  D=Fe
# ---------------------------------------------------------------------------
VERTICES = np.array([
    [0.0,              0.0,             0.0        ],   # A  TiC
    [1.0,              0.0,             0.0        ],   # B  NbC
    [0.5,  np.sqrt(3) / 2,             0.0        ],   # C  Ni
    [0.5,  np.sqrt(3) / 6,  np.sqrt(6) / 3       ],   # D  Fe
])

CORNER_NAMES = ["TiC wt%", "NbC wt%", "Ni 0–50 wt%", "Fe 0–50 wt%"]

# Max wt% each axis represents (TiC/NbC full range, Ni/Fe zoomed to 50%)
AXIS_MAX = np.array([100.0, 100.0, 50.0, 50.0])

EDGES = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
FACES = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]

# Gray → red colorscale anchored so that gray = ~100 HV
GRAY_TO_RED = [
    [0.0,  "rgb(160,160,160)"],   # gray  (low HV end)
    [0.3,  "rgb(220,180,140)"],
    [0.6,  "rgb(230,100, 60)"],
    [1.0,  "rgb(200,  0,  0)"],   # deep red  (max HV)
]


def bary_to_cart(fracs: np.ndarray) -> np.ndarray:
    """Nx4 fractional compositions → Nx3 Cartesian."""
    return fracs @ VERTICES


# ---------------------------------------------------------------------------
# Load JSON dataset
# ---------------------------------------------------------------------------
def load_json(path: str):
    with open(path) as f:
        data = json.load(f)

    comps, hv_vals, labels, hover_extra = [], [], [], []

    for m in data["materials"]:
        c = m["composition"]
        tic  = float(c.get("wt_TiC", 0) or 0)
        nbc  = float(c.get("wt_NbC", 0) or 0)
        ni   = float(c.get("wt_Ni",  0) or 0)
        fe   = float(c.get("wt_Fe",  0) or 0)
        total = tic + nbc + ni + fe
        if total == 0:
            continue
        # Scale each component by its axis max, then renormalise so the
        # Ni/Fe corners represent 50 wt% (zoom those axes by 2×)
        scaled = np.array([tic, nbc, ni, fe]) / AXIS_MAX
        scaled /= scaled.sum()
        comps.append(scaled.tolist())

        hv = m["properties"].get("hardness_HV") or 0
        hv_vals.append(float(hv))

        labels.append(m.get("label", m["id"]))

        p = m["properties"]
        src = m.get("source", {})
        kic = p.get("KIC")
        kic_str = f"{kic:.1f} MPa·m⁰·⁵" if kic else "n/a"
        hv_std = p.get("hardness_HV_std")
        hv_std_str = f" ± {hv_std:.0f}" if hv_std else ""
        author = src.get("author", "")
        year   = src.get("year", "")
        hover_extra.append(
            f"HV = {hv:.0f}{hv_std_str}<br>"
            f"KIC = {kic_str}<br>"
            f"TiC {tic:.1f}  NbC {nbc:.1f}  Ni {ni:.1f}  Fe {fe:.1f} wt%<br>"
            f"Source: {author} {year}"
        )

    return (
        np.array(comps),
        np.array(hv_vals),
        labels,
        hover_extra,
    )


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
def make_figure(comps, hv_vals, labels, hover_extra, output="tetrahedron_plot.html"):
    pts = bary_to_cart(comps)
    traces = []

    # --- translucent faces ---
    face_colors = ["#b0c4de", "#c8e6c9", "#ffe0b2", "#e1bee7"]
    for (i, j, k), color in zip(FACES, face_colors):
        tri = VERTICES[[i, j, k]]
        traces.append(go.Mesh3d(
            x=tri[:,0], y=tri[:,1], z=tri[:,2],
            i=[0], j=[1], k=[2],
            color=color, opacity=0.08,
            showscale=False, hoverinfo="skip", name="",
        ))

    # --- edges ---
    ex, ey, ez = [], [], []
    for i, j in EDGES:
        ex += [VERTICES[i,0], VERTICES[j,0], None]
        ey += [VERTICES[i,1], VERTICES[j,1], None]
        ez += [VERTICES[i,2], VERTICES[j,2], None]
    traces.append(go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode="lines",
        line=dict(color="#555", width=2),
        hoverinfo="skip", showlegend=False, name="",
    ))

    # --- tick marks at 25 / 50 / 75 % along each edge ---
    tx, ty, tz = [], [], []
    for i, j in EDGES:
        for f in (0.25, 0.50, 0.75):
            m = VERTICES[i] + f * (VERTICES[j] - VERTICES[i])
            tx.append(m[0]); ty.append(m[1]); tz.append(m[2])
    traces.append(go.Scatter3d(
        x=tx, y=ty, z=tz,
        mode="markers",
        marker=dict(size=3, color="#999"),
        hoverinfo="skip", showlegend=False, name="",
    ))

    # --- corner labels ---
    offsets = np.array([
        [-0.07, -0.07,  0.00],   # TiC  — bottom left
        [ 0.07, -0.07,  0.00],   # NbC  — bottom right
        [ 0.00,  0.07,  0.00],   # Ni   — top front
        [ 0.00, -0.04,  0.06],   # Fe   — apex
    ])
    lx = VERTICES[:,0] + offsets[:,0]
    ly = VERTICES[:,1] + offsets[:,1]
    lz = VERTICES[:,2] + offsets[:,2]
    traces.append(go.Scatter3d(
        x=lx, y=ly, z=lz,
        mode="text",
        text=CORNER_NAMES,
        textfont=dict(size=15, color="black", family="Arial Black"),
        hoverinfo="skip", showlegend=False, name="",
    ))

    # --- data points ---
    hv_min = max(100.0, float(hv_vals.min()))
    hv_max = float(hv_vals.max())

    hover_text = [
        f"<b>{lbl}</b><br>{extra}"
        for lbl, extra in zip(labels, hover_extra)
    ]

    traces.append(go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        mode="markers",
        marker=dict(
            size=8,
            color=hv_vals,
            cmin=hv_min,
            cmax=hv_max,
            colorscale=GRAY_TO_RED,
            showscale=True,
            colorbar=dict(
                title=dict(text="HV", side="right"),
                thickness=18, len=0.65,
                tickformat=".0f",
            ),
            line=dict(width=0.5, color="#333"),
        ),
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        name="samples",
    ))

    layout = go.Layout(
        title=dict(
            text="TiC – NbC – Ni – Fe  |  Hardness (HV)",
            x=0.5, font=dict(size=18),
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="white",
            camera=dict(eye=dict(x=1.7, y=1.4, z=0.8)),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font_size=13),
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.write_html(output, include_plotlyjs="cdn")
    print(f"Saved → {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "materials_dataset.json")
    output    = os.path.join(script_dir, "tetrahedron_plot.html")

    comps, hv_vals, labels, hover_extra = load_json(json_path)
    print(f"Loaded {len(comps)} samples from {json_path}")
    print(f"HV range: {hv_vals.min():.0f} – {hv_vals.max():.0f}")

    make_figure(comps, hv_vals, labels, hover_extra, output=output)

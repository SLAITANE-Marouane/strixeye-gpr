"""Reusable Streamlit UI components and plotters for the GPR app."""

import base64
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import GPR_NT, GPR_NX, GPR_DT, GPR_DX
from src.model import list_compute_devices, get_device
from app.state import Keys, ensure, set_device, PIPELINE_STEPS
from app.i18n import _, get_language

# Match figures to the StrixEyE dark theme.
plt.style.use('dark_background')
PLOTLY_TEMPLATE = 'plotly_dark'


# ------------------------------------------------------------------
# Branding
# ------------------------------------------------------------------
ASSETS = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS / "logo.png"
ICON_PATH = ASSETS / "icon.png"

ACCENT = "#6ED6DC"


def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def inject_css():
    """Global StrixEyE dark-theme styling."""
    st.markdown(f"""
<style>
    /* tighter top padding */
    .block-container {{ padding-top: 1.2rem; }}

    /* brand header banner */
    .brand-banner {{
        display: flex; align-items: center; gap: 1.6rem;
        background: linear-gradient(135deg, #0B0E13 0%, #12202A 60%, #16303A 100%);
        border: 1px solid #1E2A33; border-radius: 14px;
        padding: 1.1rem 1.6rem; margin-bottom: 0.6rem;
    }}
    .brand-banner img {{ height: 72px; }}
    .brand-title {{ font-size: 1.55rem; font-weight: 700; color: #E8EDF2;
                    letter-spacing: 0.04em; margin: 0; }}
    .brand-title span {{ color: {ACCENT}; }}
    .brand-sub {{ color: #8FA3B0; font-size: 0.86rem; margin: 0.15rem 0 0 0;
                  letter-spacing: 0.12em; text-transform: uppercase; }}

    /* pipeline status chips */
    .chip {{
        display: inline-block; padding: 0.28rem 0.7rem; margin: 0.12rem;
        border-radius: 999px; font-size: 0.78rem; font-weight: 600;
        border: 1px solid #2A3540; color: #7E8B96; background: #12161D;
        white-space: nowrap;
    }}
    .chip-done {{ color: #062A2C; background: {ACCENT}; border-color: {ACCENT}; }}
    .chip-stale {{ color: #FFD98A; border-color: #8A6D2F; background: #221C0E; }}

    /* metric cards */
    .metric-card {{
        background: #12161D; border: 1px solid #1E2A33; border-radius: 12px;
        padding: 0.9rem 1.1rem; text-align: center;
    }}
    .metric-card .val {{ font-size: 1.5rem; font-weight: 700; color: {ACCENT}; }}
    .metric-card .lbl {{ font-size: 0.78rem; color: #8FA3B0;
                         text-transform: uppercase; letter-spacing: 0.08em; }}

    /* primary buttons */
    div.stButton > button[kind="primary"] {{
        background: {ACCENT}; color: #062A2C; border: none; font-weight: 700;
    }}
    div.stButton > button[kind="primary"]:hover {{
        background: #8FE3E8; color: #062A2C;
    }}

    /* tabs */
    button[data-baseweb="tab"] {{ font-size: 0.95rem; font-weight: 600; }}

    /* sidebar logo */
    [data-testid="stSidebar"] img {{ border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)


def render_header():
    """Brand banner with logo, title and tagline."""
    st.markdown(f"""
<div class="brand-banner">
    <img src="data:image/png;base64,{_b64(LOGO_PATH)}" alt="StrixEyE logo">
    <div>
        <p class="brand-title">{_("AI-Based GPR Tunnel Detection <span>&amp; 3-D Digital Twin</span>")}</p>
        <p class="brand-sub">Eyes in the sky &middot; Intelligence on the ground</p>
    </div>
</div>
""", unsafe_allow_html=True)


def pipeline_status_bar(stale=False):
    """Row of chips showing which pipeline steps have results."""
    chips = []
    for name, keys in PIPELINE_STEPS:
        done = all(k in st.session_state for k in keys[:1])
        cls = "chip chip-done" if done else "chip"
        if stale and done:
            cls = "chip chip-stale"
        mark = "\u2713 " if done else ""
        chips.append(f'<span class="{cls}">{mark}{_(name)}</span>')
    st.markdown("\n".join(chips), unsafe_allow_html=True)


def metric_cards(items, cols=4):
    """Render (label, value) pairs as branded metric cards."""
    columns = st.columns(cols)
    for i, (label, value) in enumerate(items):
        with columns[i % cols]:
            st.markdown(
                f'<div class="metric-card"><div class="val">{value}</div>'
                f'<div class="lbl">{label}</div></div>',
                unsafe_allow_html=True,
            )


# ------------------------------------------------------------------
# Explanation panels (dual language)
# ------------------------------------------------------------------
EXPLANATIONS_EN = {
    "scene": """
**Step 1 – Scene & trajectory setup**

Define the survey area, the hidden tunnels, and the drone's flight
pattern. Each flight line produces one radar cross-section; lines
passing over a tunnel see a characteristic arc-shaped echo.
""",
    "bscan": """
**Step 2 – B-scan generation**

Every flight line becomes one radargram. The simulator emits a radar
pulse, listens for echoes from soil layers, clutter, and tunnels, and
adds realistic noise. Bright arcs are tunnel echoes; flat bands are
soil layers and the surface echo.
""",
    "preprocess": """
**Step 3 – Preprocessing**

Raw radar data are cleaned before analysis: the signal is centred and
time-aligned, flat soil-layer echoes are removed, and deep weak echoes
are amplified. This makes tunnel signatures stand out consistently for
the AI.
""",
    "ai": """
**Step 4 – AI detection**

A compact neural network (U-Net) highlights the pixels that look like
tunnel echoes. The right panel shows its confidence; the green contour
marks the detected signature.
""",
    "localize": """
**Step 5 – Localization**

Each detected signature is turned into physical estimates: the tunnel's
lateral position, its depth, and the speed of radar waves in the soil —
the three numbers that matter for mapping.
""",
    "heatmap": """
**Step 6 – Probability heat map**

All flight lines are merged into a single 2-D confidence map of the
surveyed area. Use the threshold slider to balance missed detections
against false alarms; the cyan outline is the true tunnel footprint.
""",
    "twin": """
**Step 7 – 3-D digital twin**

Detections belonging to the same tunnel are linked across lines, and a
robust axis fit turns them into a 3-D cylinder — the digital twin. The
radius cannot be measured from radar alone, so it is a user prior.
Matched pairs share colours; the residual plot shows per-line fit
quality.
""",
    "report": """
**Step 8 – Summary report**

All quality metrics in one place: map accuracy (IoU), detection
coverage, position/depth errors, and timing. Download them as JSON or
CSV.
""",
}

EXPLANATIONS_FR = {
    "scene": """
**Étape 1 – Configuration de la scène et de la trajectoire**

Définissez la zone de relevé, les tunnels cachés et le motif de vol du
drone. Chaque ligne de vol produit une coupe radar ; les lignes
passant au-dessus d'un tunnel voient un écho caractéristique en forme
d'arc.
""",
    "bscan": """
**Étape 2 – Génération des B-scans**

Chaque ligne de vol devient un radargramme. Le simulateur émet une
impulsion radar, écoute les échos des couches de sol, du fouillis et
des tunnels, et ajoute un bruit réaliste. Les arcs brillants sont les
échos des tunnels ; les bandes plates sont les couches de sol et
l'écho de surface.
""",
    "preprocess": """
**Étape 3 – Prétraitement**

Les données radar brutes sont nettoyées avant analyse : le signal est
centré et aligné en temps, les échos des couches planes sont supprimés
et les échos profonds et faibles sont amplifiés. Les signatures des
tunnels ressortent ainsi de façon cohérente pour l'IA.
""",
    "ai": """
**Étape 4 – Détection IA**

Un réseau de neurones compact (U-Net) surligne les pixels qui
ressemblent à des échos de tunnel. Le panneau de droite montre son
niveau de confiance ; le contour vert marque la signature détectée.
""",
    "localize": """
**Étape 5 – Localisation**

Chaque signature détectée est convertie en estimations physiques : la
position latérale du tunnel, sa profondeur et la vitesse des ondes
radar dans le sol — les trois nombres qui comptent pour la cartographie.
""",
    "heatmap": """
**Étape 6 – Carte de chaleur de probabilité**

Toutes les lignes de vol sont fusionnées en une seule carte de
confiance 2-D de la zone survolée. Utilisez le curseur de seuil pour
arbitrer entre détections manquées et fausses alarmes ; le contour cyan
est l'empreinte réelle du tunnel.
""",
    "twin": """
**Étape 7 – Jumeau numérique 3-D**

Les détections appartenant au même tunnel sont reliées entre les
lignes, et un ajustement robuste de l'axe les transforme en cylindre
3-D — le jumeau numérique. Le rayon ne pouvant pas être mesuré par le
radar seul, il s'agit d'un a priori utilisateur. Les paires appariées
partagent les couleurs ; le graphique des résidus montre la qualité de
l'ajustement par ligne.
""",
    "report": """
**Étape 8 – Rapport de synthèse**

Toutes les métriques de qualité au même endroit : précision de la
carte (IoU), couverture des détections, erreurs de position/profondeur
et chronométrage. Téléchargez-les en JSON ou CSV.
""",
}

EXPLANATIONS = {"en": EXPLANATIONS_EN, "fr": EXPLANATIONS_FR}


def explanation(step_key):
    lang = EXPLANATIONS.get(get_language(), EXPLANATIONS_EN)
    st.markdown(lang.get(step_key, ""))


# ------------------------------------------------------------------
# Startup tutorial (dual language)
# ------------------------------------------------------------------
TUTORIAL_PAGES_EN = [
    {
        "title": "Welcome to StrixEyE GPR",
        "body": """
Simulate a drone radar survey, let the AI find hidden tunnels, and
rebuild them in 3-D — all on your own machine.

The workflow has 8 guided steps, from scene setup to a downloadable
report. It takes about a minute to run.
""",
    },
    {
        "title": "1. Set up your scene",
        "body": """
In the **sidebar**:

- Pick the **survey size** and flight pattern.
- Add **tunnels** (or hit **🎲 Random scenario**).
- Pin the **random seed** if you want reproducible results.

The plan view in **Step 1** updates live.
""",
    },
    {
        "title": "2. Run the pipeline",
        "body": """
- Easiest: press **▶ Run full pipeline** in the sidebar — every tab
  fills in automatically.
- Or walk through the tabs one by one; each has its own run button.

A trained model is required (`python -m scripts.train`) — the app loads
the newest checkpoint in `models/` automatically.
""",
    },
    {
        "title": "3. Read the results",
        "body": """
- **Heat map** (Step 6): tune the threshold, compare with the true
  footprint.
- **3-D twin** (Step 7): rotate it; check the per-tunnel errors and
  residual plot.
- **Report** (Step 8): download JSON / CSV.

Reopen this guide anytime with **📖 Show tutorial**.
""",
    },
]

TUTORIAL_PAGES_FR = [
    {
        "title": "Bienvenue dans StrixEyE GPR",
        "body": """
Simulez un relevé radar par drone, laissez l'IA trouver les tunnels
cachés et reconstruisez-les en 3-D — le tout sur votre propre machine.

Le workflow comporte 8 étapes guidées, de la configuration de la scène
au rapport téléchargeable. Environ une minute d'exécution.
""",
    },
    {
        "title": "1. Configurez votre scène",
        "body": """
Dans la **barre latérale** :

- Choisissez la **taille du relevé** et le motif de vol.
- Ajoutez des **tunnels** (ou cliquez sur **🎲 Scénario aléatoire**).
- Fixez la **graine aléatoire** pour des résultats reproductibles.

La vue plan de l'**étape 1** se met à jour en direct.
""",
    },
    {
        "title": "2. Exécutez le pipeline",
        "body": """
- Le plus simple : cliquez sur **▶ Exécuter tout le pipeline** dans la
  barre latérale — chaque onglet se remplit automatiquement.
- Ou parcourez les onglets un par un ; chacun a son propre bouton.

Un modèle entraîné est requis (`python -m scripts.train`) —
l'application charge automatiquement le checkpoint le plus récent dans
`models/`.
""",
    },
    {
        "title": "3. Lisez les résultats",
        "body": """
- **Carte de chaleur** (étape 6) : ajustez le seuil, comparez avec
  l'empreinte réelle.
- **Jumeau 3-D** (étape 7) : faites-le pivoter ; consultez les erreurs
  par tunnel et le graphique des résidus.
- **Rapport** (étape 8) : téléchargez JSON / CSV.

Rouvrez ce guide à tout moment avec **📖 Afficher le tutoriel**.
""",
    },
]

TUTORIAL_PAGES = {"en": TUTORIAL_PAGES_EN, "fr": TUTORIAL_PAGES_FR}


@st.dialog("Getting started / Pour commencer", width="large")
def show_tutorial():
    """Modal startup tutorial with previous/next navigation."""
    page_key = "tutorial_page"
    page = st.session_state.get(page_key, 0)
    pages = TUTORIAL_PAGES.get(get_language(), TUTORIAL_PAGES_EN)
    n = len(pages)

    if page == 0 and LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=320)
    st.markdown(f"### {pages[page]['title']}")
    st.markdown(pages[page]["body"])

    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        if st.button(_("← Previous"), disabled=page == 0, width='stretch'):
            st.session_state[page_key] = max(0, page - 1)
            # Re-arm the one-shot trigger so the dialog survives the rerun.
            st.session_state["tutorial_trigger"] = True
            st.rerun()
    with c3:
        if page < n - 1:
            if st.button(_("Next →"), width='stretch'):
                st.session_state[page_key] = min(n - 1, page + 1)
                st.session_state["tutorial_trigger"] = True
                st.rerun()
        else:
            if st.button(_("Get started"), width='stretch'):
                # No re-arm: the dialog closes and stays closed.
                st.session_state[page_key] = 0
                st.rerun()
    st.caption(_("Page {p} of {n}").format(p=page + 1, n=n))


# ------------------------------------------------------------------
# Device selector
# ------------------------------------------------------------------
def device_selector():
    st.sidebar.header(_("Compute device"))
    devices = list_compute_devices()
    gpu_available = any(d[0].startswith("cuda") for d in devices)

    if not gpu_available:
        st.sidebar.info(_(
            "No CUDA GPU detected. The app will run on CPU.\n\n"
            "Training and batch inference will be slower; all functionality "
            "remains available."
        ))
        return get_device()

    options = {label: name for name, label in devices}
    default = devices[1][1] if gpu_available else devices[0][1]
    choice = st.sidebar.selectbox(
        _("GPU detected. Select device (GPU recommended):"),
        list(options.keys()),
        index=list(options.keys()).index(default),
        help=_("Applies to U-Net inference and all batched computations."),
    )
    dev_name = options[choice]
    if ensure(Keys.DEVICE, "cpu") != dev_name or not ensure(Keys.DEVICE_CONFIRMED, False):
        set_device(dev_name)
    st.sidebar.caption(_("Active device: **{choice}**").format(choice=choice))
    return get_device()


def device_badge(device):
    """Return a short human-readable device string for status messages."""
    name = str(device)
    if name.startswith("cuda"):
        import torch
        return torch.cuda.get_device_name(int(name.split(":")[-1]))
    return "CPU"


# ------------------------------------------------------------------
# Trajectory generator
# ------------------------------------------------------------------
def generate_trajectory(width, length, spacing, pattern="lawnmower"):
    """Return list of line dicts for the chosen pattern.

    Each dict has keys ``y`` (line coordinate), ``x0``, ``x1`` and
    ``vertical``. Vertical lines are used only for plan-view display.
    """
    n = max(1, int(round(length / spacing)))
    ys = np.linspace(0, length, n + 1)
    lines = []
    for i, y in enumerate(ys):
        x0, x1 = (0.0, width) if i % 2 == 0 else (width, 0.0)
        lines.append(dict(y=float(y), x0=x0, x1=x1, vertical=False))
    if pattern == "grid":
        nx = max(1, int(round(width / spacing)))
        xs = np.linspace(0, width, nx + 1)
        for j, x in enumerate(xs):
            y0, y1 = (0.0, length) if j % 2 == 0 else (length, 0.0)
            lines.append(dict(y=float(x), x0=y0, x1=y1, vertical=True))
    return lines


# ------------------------------------------------------------------
# Plotly figures
# ------------------------------------------------------------------
def plot_plan_view(tunnels, trajectory, width, length, pattern):
    """Interactive plan-view map of tunnels + flight lines."""
    fig = go.Figure()
    # tunnels
    for k, t in enumerate(tunnels):
        fig.add_trace(go.Scatter(
            x=[t['x0_m'], t['x1_m']],
            y=[t['y0_m'], t['y1_m']],
            mode='lines+markers',
            name=_("Tunnel {i}").format(i=k+1) + f" (r={t['radius_m']:.2f} m)",
            line=dict(width=4),
        ))
    # trajectory
    xs, ys = [], []
    for line in trajectory:
        if line['vertical']:
            xs.extend([line['y'], line['y'], None])
            ys.extend([line['x0'], line['x1'], None])
        else:
            xs.extend([line['x0'], line['x1'], None])
            ys.extend([line['y'], line['y'], None])
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='lines',
        line=dict(color='gray', width=1, dash='dot'),
        name=_("flight lines")
    ))
    fig.update_xaxes(range=[-0.5, width + 0.5], title=_("x [m]"))
    fig.update_yaxes(range=[-0.5, length + 0.5], title=_("y [m]"),
                     scaleanchor='x', scaleratio=1)
    fig.update_layout(title=_("Survey plan view ({pattern})").format(pattern=pattern),
                      height=500, legend=dict(orientation='h'),
                      template=PLOTLY_TEMPLATE,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    return fig


def plot_heatmap_interactive(heatmap, extent, xs, ys, gt_mask, tau):
    """Plotly 2-D heat map with GT overlay and threshold."""
    x_min, x_max, y_max, y_min = extent
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=heatmap,
        x=xs,
        y=ys,
        colorscale='Hot',
        zmin=0, zmax=1,
        colorbar=dict(title=_("probability")),
        name=_("probability"),
    ))
    # GT contour
    if gt_mask is not None:
        fig.add_trace(go.Contour(
            z=gt_mask.astype(int),
            x=xs,
            y=ys,
            showscale=False,
            contours=dict(start=0.5, end=0.5, coloring='none'),
            line=dict(color='cyan', width=2),
            name=_("GT footprint"),
        ))
    fig.update_xaxes(range=[x_min, x_max], title=_("x [m]"))
    fig.update_yaxes(range=[y_max, y_min], title=_("y [m]"),
                     scaleanchor='x', scaleratio=1)
    fig.update_layout(
        title=_("Probability heat map (threshold = {tau:.2f})").format(tau=tau),
                      height=520, template=PLOTLY_TEMPLATE,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    return fig


def plot_3d_twin(tunnels_truth, twins, width, length, matches=None):
    """Plotly 3-D digital twin: GT tunnels vs per-track reconstructed twins.

    ``twins`` is the list returned by ``src.twin.reconstruct_twins``;
    ``matches`` the list returned by ``src.twin.evaluate_twins`` (used to
    colour each twin like its matched ground-truth tunnel; unmatched
    twins are grey).
    """
    from plotly import colors as pc
    from src.twin import cylinder_mesh

    palette = pc.qualitative.Set2
    fig = go.Figure()

    # Ground-truth tunnels (each over its own y range).
    for k, t in enumerate(tunnels_truth):
        col = palette[k % len(palette)]
        gy0, gy1 = sorted((t['y0_m'], t['y1_m']))
        if gy1 - gy0 < 0.2:
            gy1 = gy0 + 0.2
        yc = np.linspace(gy0, gy1, 30)
        frac = (yc - t['y0_m']) / max(1e-9, t['y1_m'] - t['y0_m'])
        xc = t['x0_m'] + (t['x1_m'] - t['x0_m']) * frac
        zc = np.full_like(yc, t['depth_m'])
        xg, yg, zg = cylinder_mesh(
            lambda y, _yc=yc, _xc=xc: np.interp(y, _yc, _xc),
            lambda y, _yc=yc, _zc=zc: np.interp(y, _yc, _zc),
            gy0, gy1, t['radius_m']
        )
        fig.add_trace(go.Surface(
            x=xg, y=yg, z=zg,
            colorscale=[[0, col], [1, col]],
            showscale=False,
            opacity=0.6,
            name=_("GT tunnel {i}").format(i=k + 1)
        ))

    # Reconstructed twins (matched ones take the GT tunnel's colour).
    twin_to_gt = {}
    if matches:
        for e in matches:
            if e.get('twin'):
                twin_to_gt[e['twin'] - 1] = e['tunnel'] - 1
    for wi, tw in enumerate(twins or []):
        gt_idx = twin_to_gt.get(wi)
        col = palette[gt_idx % len(palette)] if gt_idx is not None else 'grey'
        xr, yr, zr = cylinder_mesh(tw['x_fn'], tw['z_fn'],
                                   tw['y0'], tw['y1'], tw['radius'])
        fig.add_trace(go.Surface(
            x=xr, y=yr, z=zr,
            colorscale=[[0, col], [1, col]],
            showscale=False,
            opacity=0.4,
            name=_("Twin {i}").format(i=wi + 1)
        ))
        fig.add_trace(go.Scatter3d(
            x=tw['xs'], y=tw['ys'], z=tw['zs'],
            mode='markers',
            marker=dict(size=3, color=col, line=dict(width=1, color='black')),
            name=_("per-line apex") + f" {wi + 1}"
        ))

    # Soil volume wireframe
    max_depth = 4.0
    corners = [
        [0, 0, 0], [width, 0, 0], [width, length, 0], [0, length, 0],
        [0, 0, max_depth], [width, 0, max_depth], [width, length, max_depth], [0, length, max_depth]
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for a, b in edges:
        fig.add_trace(go.Scatter3d(
            x=[corners[a][0], corners[b][0]],
            y=[corners[a][1], corners[b][1]],
            z=[corners[a][2], corners[b][2]],
            mode='lines',
            line=dict(color='saddlebrown', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title=_("x [m]"), range=[0, width]),
            yaxis=dict(title=_("y [m]"), range=[0, length]),
            zaxis=dict(title=_("depth [m]"), range=[max_depth, 0]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=length / width, z=0.4)
        ),
        height=600,
        legend=dict(orientation='h'),
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def plot_residuals(twins):
    """Per-line lateral/depth residuals of the robust axis fits."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for k, tw in enumerate(twins):
        lbl = _("Twin {i}").format(i=k + 1)
        axes[0].plot(tw['ys'], tw['res_x'], 'o-', ms=3, lw=1, label=lbl)
        axes[1].plot(tw['ys'], tw['res_z'], 'o-', ms=3, lw=1, label=lbl)
    for ax in axes:
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_xlabel(_("Line y [m]"))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(_("Lateral residual [m]"))
    axes[1].set_ylabel(_("Depth residual [m]"))
    axes[0].legend()
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Matplotlib helpers
# ------------------------------------------------------------------
def plot_bscan(ax, img, title="", mask=None, dets=None, gt_params=None):
    """Render a B-scan with optional mask / detections / GT overlay."""
    vmax = np.abs(img).max()
    ax.imshow(img, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax,
              extent=[0, GPR_NX * GPR_DX, GPR_NT * GPR_DT, 0])
    if mask is not None:
        ax.contour(mask, colors='lime', linewidths=1.0,
                   extent=[0, GPR_NX * GPR_DX, GPR_NT * GPR_DT, 0])
    if gt_params is not None:
        xg = np.arange(GPR_NX) * GPR_DX
        for g in gt_params:
            t = np.sqrt(np.maximum(g['t0_ns'] ** 2 + 4 * (xg - g['x0_m']) ** 2 / g['v'] ** 2, 0))
            ax.plot(xg, t, 'g--', lw=1.5)
    if dets is not None:
        xg = np.arange(GPR_NX) * GPR_DX
        for d in dets:
            t = np.sqrt(np.maximum(d['t0_ns'] ** 2 + 4 * (xg - d['x0_m']) ** 2 / d['v'] ** 2, 0))
            ax.plot(xg, t, 'r-', lw=1.5)
    ax.set_title(title)
    ax.set_xlabel(_("x [m]"))
    ax.set_ylabel(_("two-way time [ns]"))


# ------------------------------------------------------------------
# Export helpers
# ------------------------------------------------------------------
def export_report(metrics, path="outputs/report.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)


def report_dataframe(metrics):
    return pd.DataFrame([metrics])


def build_pdf_report(metrics, config, heatmap=None, extent=None):
    """One-page branded PDF report (A4 portrait); returns PDF bytes.

    Layout: StrixEyE logo + title, configuration summary, metrics table,
    and the detection heat map when available. Rendered with matplotlib
    (no extra dependencies) on a clean light style.
    """
    import io
    from datetime import datetime
    from matplotlib import image as mpimg

    with plt.style.context('default'):  # light page regardless of app theme
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')

        # --- logo + title -------------------------------------------------
        if LOGO_PATH.exists():
            ax_logo = fig.add_axes([0.07, 0.90, 0.32, 0.075])
            ax_logo.imshow(mpimg.imread(LOGO_PATH))
            ax_logo.axis('off')
        fig.text(0.07, 0.862,
                 _("AI-Based GPR Tunnel Detection <span>&amp; 3-D Digital Twin</span>")
                 .replace("<span>&amp;", "&").replace("</span>", ""),
                 fontsize=15, fontweight='bold', color='#0B0E13')
        fig.text(0.07, 0.845,
                 _("Summary report") + " — " + datetime.now().strftime("%Y-%m-%d %H:%M"),
                 fontsize=10, color='#55606A')

        # --- configuration -------------------------------------------------
        cfg_lines = [
            f"{_("Survey")}: {config['width']:.1f} × {config['length']:.1f} m"
            f"   ·   {_("Line spacing [m]")}: {config['spacing']} m"
            f"   ·   {_("Pattern")}: {config['pattern']}",
            f"{_("Tunnels")}: {config['n_tunnels']}"
            f"   ·   {_("Seed")}: {config['seed']}"
            f"   ·   {_("Device")}: {config['device']}",
        ]
        fig.text(0.07, 0.812, "\n".join(cfg_lines), fontsize=9.5,
                 color='#0B0E13', va='top', family='monospace')

        # --- metrics table -------------------------------------------------
        rows = []

        def _add(label, value, fmt="{:.3f}"):
            if value is not None:
                rows.append((label, fmt.format(value) if isinstance(value, float)
                             else str(value)))

        _add(_("Plan-view IoU"), metrics.get('plan_view_iou'))
        _add(_("Best threshold"), metrics.get('best_threshold'), "{:.2f}")
        if metrics.get('lines_with_detections') is not None:
            rows.append((_("Lines w/ detections"),
                         f"{metrics['lines_with_detections']}/{metrics['total_lines']}"))
        if metrics.get('tunnels_matched') is not None:
            rows.append((_("Tunnels matched"),
                         f"{metrics['tunnels_matched']}/{metrics['tunnels_total']}"))
        _add(_("Mean coverage"), metrics.get('mean_coverage'), "{:.0%}")
        _add(_("Lateral MAE") + " [m]", metrics.get('lateral_mae'))
        _add(_("Depth MAE") + " [m]", metrics.get('depth_mae'))
        _add(_("Velocity error") + " [m/ns]", metrics.get('v_err'))
        _add(_("Inference time") + " [s]", metrics.get('inference_time_s'), "{:.2f}")

        ax_tbl = fig.add_axes([0.07, 0.56, 0.5, 0.22])
        ax_tbl.axis('off')
        tbl = ax_tbl.table(cellText=rows, colLabels=[_("Metric"), _("Value")],
                           cellLoc='left', loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor('#C9D2DA')
            if r == 0:
                cell.set_text_props(fontweight='bold', color='white')
                cell.set_facecolor('#16303A')

        # --- heat map ------------------------------------------------------
        if heatmap is not None and extent is not None:
            x_min, x_max, y_max, y_min = extent
            ax_hm = fig.add_axes([0.10, 0.16, 0.62, 0.34])
            im = ax_hm.imshow(heatmap, cmap='hot', aspect='auto', vmin=0, vmax=1,
                              extent=[x_min, x_max, y_min, y_max])
            ax_hm.set_title(_("Detection probability heat map"), fontsize=10)
            ax_hm.set_xlabel(_("x [m]"))
            ax_hm.set_ylabel(_("y [m]"))
            cb = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
            cb.set_label(_("probability"), fontsize=8)

        # --- footer --------------------------------------------------------
        fig.text(0.07, 0.045,
                 _("Generated by StrixEyE — AI-GPR Tunnel Detection"),
                 fontsize=8, color='#8FA3B0')
        fig.text(0.07, 0.030,
                 _("Eyes in the sky · Intelligence on the ground"),
                 fontsize=7.5, color='#B8C2CB', style='italic')

        buf = io.BytesIO()
        fig.savefig(buf, format='pdf', facecolor='white')
        plt.close(fig)
        return buf.getvalue()

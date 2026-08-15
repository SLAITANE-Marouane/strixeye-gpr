"""Minimal i18n layer for the StrixEyE GPR app (English / Français).

Usage
-----
Wrap any user-facing string with :func:`_`::

    from app.i18n import _
    st.button(_("Generate B-scans"))
    st.success(_("Generated {n} B-scans.").format(n=n))

Long-form content (explanations, tutorial pages) lives in dual-language
dictionaries inside ``app.components``; short UI strings are translated
through :data:`STRINGS_FR` below. Unknown keys fall back to English.
"""

import streamlit as st

#: Display label -> language code.
LANGUAGES = {"English": "en", "Français": "fr"}


def get_language():
    """Current language code ('en' or 'fr')."""
    return st.session_state.get("language", "en")


def language_selector():
    """Sidebar language dropdown; stores the choice in session state."""
    if "lang_widget" not in st.session_state:
        st.session_state["lang_widget"] = "English"
    choice = st.sidebar.selectbox("🌐 Language / Langue", list(LANGUAGES.keys()),
                                  key="lang_widget")
    st.session_state["language"] = LANGUAGES[choice]


def _(text):
    """Translate a short UI string into the active language."""
    if get_language() == "fr":
        return STRINGS_FR.get(text, text)
    return text


# ------------------------------------------------------------------
# French translations of short UI strings (key = English source)
# ------------------------------------------------------------------
STRINGS_FR = {
    # sidebar
    "Control Panel": "Panneau de contrôle",
    "📖 Show tutorial": "📖 Afficher le tutoriel",
    "Compute device": "Dispositif de calcul",
    "No CUDA GPU detected. The app will run on CPU.\n\n"
    "Training and batch inference will be slower; all functionality "
    "remains available.":
        "Aucun GPU CUDA détecté. L'application utilisera le CPU.\n\n"
        "L'entraînement et l'inférence par lots seront plus lents ; "
        "toutes les fonctionnalités restent disponibles.",
    "GPU detected. Select device (GPU recommended):":
        "GPU détecté. Sélectionnez le dispositif (GPU recommandé) :",
    "Applies to U-Net inference and all batched computations.":
        "S'applique à l'inférence U-Net et à tous les calculs par lots.",
    "Active device: **{choice}**": "Dispositif actif : **{choice}**",
    "Reproducibility": "Reproductibilité",
    "Global random seed": "Graine aléatoire globale",
    "Set seed": "Fixer la graine",
    "Seeds fixed.": "Graines fixées.",
    "Survey configuration": "Configuration du relevé",
    "Survey width [m]": "Largeur du relevé [m]",
    "Survey length [m]": "Longueur du relevé [m]",
    "Line spacing [m]": "Espacement des lignes [m]",
    "Trace spacing [m]": "Espacement des traces [m]",
    "Fixed by model input size ({n} traces -> {d:.2f} m line).":
        "Fixé par la taille d'entrée du modèle ({n} traces -> {d:.2f} m par ligne).",
    "Range 2.0–{m:.2f} m. Capped by the model input: one B-scan covers "
    "{m:.2f} m ({n} traces × {d} m).":
        "Plage 2,0–{m:.2f} m. Limité par l'entrée du modèle : un B-scan couvre "
        "{m:.2f} m ({n} traces × {d} m).",
    "Range 2–50 m. Number of flight lines ≈ length ÷ line spacing.":
        "Plage 2–50 m. Nombre de lignes de vol ≈ longueur ÷ espacement des lignes.",
    "Distance between adjacent flight lines. Smaller spacing = denser "
    "survey and more B-scans to process.":
        "Distance entre lignes de vol adjacentes. Un espacement plus petit = "
        "relevé plus dense et plus de B-scans à traiter.",
    "Limits: width ≤ {w:.2f} m (model line length) · length ≤ 50 m · "
    "up to 5 tunnels.":
        "Limites : largeur ≤ {w:.2f} m (longueur de ligne du modèle) · "
        "longueur ≤ 50 m · jusqu'à 5 tunnels.",
    "0–5 tunnels. Each tunnel adds its start/end points, depth, radius "
    "and soil velocity.":
        "0–5 tunnels. Chaque tunnel ajoute ses points de début/fin, sa "
        "profondeur, son rayon et la vitesse du sol.",
    "Ranges — depth 0.5–4 m · radius 0.1–2 m · soil velocity 0.05–0.20 m/ns. "
    "Coordinates must lie inside the survey.":
        "Plages — profondeur 0,5–4 m · rayon 0,1–2 m · vitesse du sol "
        "0,05–0,20 m/ns. Les coordonnées doivent être à l'intérieur du relevé.",
    "Trajectory pattern": "Motif de trajectoire",
    "lawnmower": "tondeuse",
    "grid": "grille",
    "Lawnmower: the drone flies parallel lines back and forth along x, "
    "stepping in y after each pass (one B-scan per line).\n\n"
    "Grid: same lawnmower scan plus a second set of perpendicular lines "
    "(shown in the plan view) for cross-line confirmation.":
        "Tondeuse : le drone parcourt des lignes parallèles en va-et-vient "
        "le long de x, avec un pas en y après chaque passage (un B-scan par "
        "ligne).\n\nGrille : même balayage en tondeuse plus un second jeu de "
        "lignes perpendiculaires (visibles sur la vue plan) pour une "
        "confirmation croisée.",
    "Fixes every random generator (tunnel layout, simulator noise, clutter) "
    "so the whole pipeline is reproducible. Click 'Set seed' to apply it; "
    "change the value to explore a different random realisation.":
        "Fixe tous les générateurs aléatoires (disposition des tunnels, bruit "
        "du simulateur, fouillis) pour rendre tout le pipeline reproductible. "
        "Cliquez sur 'Fixer la graine' pour l'appliquer ; changez la valeur "
        "pour explorer une autre réalisation aléatoire.",
    "Ground-truth tunnels": "Tunnels de vérité terrain",
    "Number of tunnels": "Nombre de tunnels",
    "Tunnel {i}": "Tunnel {i}",
    "Start x (m)": "Début x (m)",
    "Start y (m)": "Début y (m)",
    "End x (m)": "Fin x (m)",
    "End y (m)": "Fin y (m)",
    "Depth (m)": "Profondeur (m)",
    "Radius (m)": "Rayon (m)",
    "Soil velocity (m/ns)": "Vitesse du sol (m/ns)",
    "🎲 Random scenario": "🎲 Scénario aléatoire",
    "Run pipeline": "Exécution du pipeline",
    "▶ Run full pipeline": "▶ Exécuter tout le pipeline",
    "Scene configuration changed — previous results were cleared.":
        "Configuration de la scène modifiée — les résultats précédents ont été effacés.",

    # tabs
    "1. Scene setup": "1. Configuration de la scène",
    "2. B-scans": "2. B-scans",
    "3. Preprocessing": "3. Prétraitement",
    "4. AI detection": "4. Détection IA",
    "5. Localization": "5. Localisation",
    "6. Heat map": "6. Carte de chaleur",
    "7. 3-D twin": "7. Jumeau 3-D",
    "8. Report": "8. Rapport",

    # pipeline status chips
    "B-scans": "B-scans",
    "Preprocessing": "Prétraitement",
    "AI detection": "Détection IA",
    "Localization": "Localisation",
    "Heat map": "Carte de chaleur",
    "3-D twin": "Jumeau 3-D",
    "Report": "Rapport",

    # model loading
    "Loading U-Net checkpoint ...": "Chargement du checkpoint U-Net ...",
    "No trained checkpoint found in `models/`. Please run `python -m scripts.train` first.":
        "Aucun checkpoint entraîné trouvé dans `models/`. "
        "Exécutez d'abord `python -m scripts.train`.",

    # full pipeline status
    "Running full pipeline ...": "Exécution du pipeline complet ...",
    "Pipeline complete ✔": "Pipeline terminé ✔",

    # step messages
    "Generated {n} B-scans.": "{n} B-scans générés.",
    "Preprocessed {n} B-scans.": "{n} B-scans prétraités.",
    "U-Net inference on {dev} — {n} lines.":
        "Inférence U-Net sur {dev} — {n} lignes.",
    "ms/line": "ms/ligne",
    "Localised detections on {n}/{m} lines.":
        "Détections localisées sur {n}/{m} lignes.",
    "Heat map built.": "Carte de chaleur construite.",
    "Digital twin reconstructed.": "Jumeau numérique reconstruit.",
    "{n} twin(s) reconstructed.": "{n} jumeau(x) reconstruit(s).",
    "No reconstruction possible (a track needs ≥ 3 detections on consecutive lines).":
        "Aucune reconstruction possible (une piste nécessite ≥ 3 "
        "détections sur des lignes consécutives).",
    "Residuals of the robust (Huber) axis fit per track — spikes mark outlier detections the fit rejected.":
        "Résidus de l'ajustement robuste (Huber) de l'axe par piste — "
        "les pics marquent les détections aberrantes rejetées par l'ajustement.",
    "Twin {i}": "Jumeau {i}",
    "Twin": "Jumeau",
    "Tunnel": "Tunnel",
    "Coverage": "Couverture",
    "Per-line residuals": "Résidus par ligne",
    "Lateral residual [m]": "Résidu latéral [m]",
    "Depth residual [m]": "Résidu de profondeur [m]",
    "Line y [m]": "Ligne y [m]",
    "GT tunnel {i}": "Tunnel GT {i}",
    "Report generated.": "Rapport généré.",

    # step buttons / spinners / warnings
    "Generate B-scans": "Générer les B-scans",
    "Simulating B-scans ...": "Simulation des B-scans ...",
    "Run preprocessing": "Lancer le prétraitement",
    "Preprocessing ...": "Prétraitement ...",
    "Run U-Net detection": "Lancer la détection U-Net",
    "U-Net inference on {dev} ...": "Inférence U-Net sur {dev} ...",
    "Localize tunnels": "Localiser les tunnels",
    "Fitting hyperbolae ...": "Ajustement des hyperboles ...",
    "Build heat map": "Construire la carte de chaleur",
    "Fusing lines ...": "Fusion des lignes ...",
    "Reconstruct 3-D twin": "Reconstruire le jumeau 3-D",
    "Reconstructing twin ...": "Reconstruction du jumeau ...",
    "Generate report": "Générer le rapport",
    "Generate B-scans first (Step 2).": "Générez d'abord les B-scans (étape 2).",
    "Preprocess B-scans first (Step 3).": "Prétraitez d'abord les B-scans (étape 3).",
    "Run AI detection first (Step 4).": "Exécutez d'abord la détection IA (étape 4).",
    "Localize tunnels first (Step 5).": "Localisez d'abord les tunnels (étape 5).",

    # step-specific widgets / captions
    "Line index": "Indice de ligne",
    "Ground-truth hyperbola mask overlaid in lime.":
        "Masque d'hyperbole de vérité terrain superposé en vert citron.",
    "Raw B-scan – line {i}": "B-scan brut – ligne {i}",
    "Preprocessed input": "Entrée prétraitée",
    "U-Net probability (≥0.5 contour in lime)":
        "Probabilité U-Net (contour ≥ 0,5 en vert citron)",
    "Localization – line {i}": "Localisation – ligne {i}",
    "Threshold": "Seuil",
    "Best threshold: **{tau:.2f}** → plan-view IoU **{iou:.3f}**":
        "Meilleur seuil : **{tau:.2f}** → IoU en vue plan **{iou:.3f}**",
    "Axis errors": "Erreurs d'axe",
    "Lateral MAE [m]": "MAE latérale [m]",
    "Depth MAE [m]": "MAE de profondeur [m]",
    "Velocity error [m/ns]": "Erreur de vitesse [m/ns]",
    "Radius bias [m]": "Biais de rayon [m]",
    "Metric": "Métrique",
    "Value": "Valeur",

    # report tab
    "Plan-view IoU": "IoU vue plan",
    "Lines w/ detections": "Lignes avec détections",
    "Lateral MAE": "MAE latérale",
    "Depth MAE": "MAE profondeur",
    "Inference time": "Temps d'inférence",
    "Device": "Dispositif",
    "Full metrics (JSON)": "Métriques complètes (JSON)",
    "⬇ Download JSON": "⬇ Télécharger JSON",
    "⬇ Download CSV": "⬇ Télécharger CSV",
    "⬇ Download PDF": "⬇ Télécharger PDF",

    # presentation mode
    "Presentation": "Présentation",
    "🎬 Presentation mode": "🎬 Mode présentation",
    "Hides all ground-truth overlays until you reveal them — perfect for live demos.":
        "Masque toutes les vérités terrain jusqu'à leur révélation — "
        "parfait pour les démos en direct.",
    "🎭 Reveal ground truth": "🎭 Révéler la vérité terrain",
    "🙈 Hide again": "🙈 Masquer à nouveau",
    "Ground truth revealed 🎭": "Vérité terrain révélée 🎭",
    "🔒 Ground truth hidden (presentation mode)":
        "🔒 Vérité terrain masquée (mode présentation)",

    # PDF report
    "Summary report": "Rapport de synthèse",
    "Survey": "Relevé",
    "Pattern": "Motif",
    "Tunnels": "Tunnels",
    "Seed": "Graine",
    "Best threshold": "Meilleur seuil",
    "Tunnels matched": "Tunnels appariés",
    "Mean coverage": "Couverture moyenne",
    "Velocity error": "Erreur de vitesse",
    "Detection probability heat map": "Carte de chaleur de probabilité de détection",
    "Generated by StrixEyE — AI-GPR Tunnel Detection":
        "Généré par StrixEyE — Détection de tunnels par GPR et IA",
    "Eyes in the sky · Intelligence on the ground":
        "Eyes in the sky · Intelligence on the ground",
    "💾 Save to outputs/": "💾 Enregistrer dans outputs/",
    "Saved outputs/report.json and outputs/report.csv":
        "outputs/report.json et outputs/report.csv enregistrés",

    # plots (components)
    "flight lines": "lignes de vol",
    "Survey plan view ({pattern})": "Vue plan du relevé ({pattern})",
    "probability": "probabilité",
    "GT footprint": "Empreinte GT",
    "Probability heat map (threshold = {tau:.2f})":
        "Carte de chaleur de probabilité (seuil = {tau:.2f})",
    "GT tunnel": "Tunnel GT",
    "Reconstructed twin": "Jumeau reconstruit",
    "per-line apex": "apex par ligne",
    "x [m]": "x [m]",
    "y [m]": "y [m]",
    "depth [m]": "profondeur [m]",
    "two-way time [ns]": "temps aller-retour [ns]",
    "raw": "brut",
    "1) dewow": "1) dewow",
    "2) time-zero": "2) temps zéro",
    "3) bg-removed": "3) fond supprimé",
    "4) SEC+AGC": "4) SEC+AGC",
    "5) normalised": "5) normalisé",

    # header banner
    "AI-Based GPR Tunnel Detection <span>&amp; 3-D Digital Twin</span>":
        "Détection de tunnels par GPR et IA <span>&amp; jumeau numérique 3-D</span>",

    # tutorial chrome
    "Getting started": "Pour commencer",
    "← Previous": "← Précédent",
    "Next →": "Suivant →",
    "Get started": "C'est parti",
    "Page {p} of {n}": "Page {p} sur {n}",
}

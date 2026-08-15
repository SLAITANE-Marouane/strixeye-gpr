"""Headless smoke test for the Streamlit app using streamlit.testing.AppTest.

Run with:
    python scripts/smoke_test_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest


def main():
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, f"App raised on load: {at.exception}"
    print("[ok] app loads without exceptions")
    print("     tabs:", [t.label for t in at.tabs])

    # Report with no pipeline results: must warn, not crash (cloud bug).
    btn = next(b for b in at.button if "Generate report" in b.label)
    btn.click().run()
    assert not at.exception, f"report without pipeline raised: {at.exception}"
    assert "report_metrics" not in at.session_state, \
        "empty report should not be stored"
    # PDF builder must also survive a metrics dict with no known keys.
    from app.components import build_pdf_report
    pdf = build_pdf_report(
        {"device": "cpu"},
        dict(width=10.0, length=10.0, spacing=0.5, pattern="lawnmower",
             n_tunnels=0, seed=42, device="cpu"))
    assert pdf[:5] == b"%PDF-", "empty-metrics PDF is not valid"
    print("[ok] report button guarded + empty-metrics PDF builds")

    # Click "Run full pipeline" in the sidebar.
    btn = next(b for b in at.button if "Run full pipeline" in b.label)
    btn.click().run()
    assert not at.exception, f"App raised during full pipeline: {at.exception}"
    print("[ok] full pipeline executed without exceptions")

    # Verify every pipeline result landed in session state.
    required = [
        "raw_scans", "gt_masks", "gt_params", "ys", "preprocessed",
        "probs", "line_dets", "heatmap", "gt_plan_mask", "twin",
        "twin_errors", "report_metrics",
    ]
    missing = [k for k in required if k not in at.session_state]
    assert not missing, f"Missing session-state keys: {missing}"
    print("[ok] all pipeline results present:", ", ".join(required))

    # Sanity: no phantom tunnels -> GT masks must be empty on lines with
    # no tunnel crossing (params list empty).
    empties = sum(1 for p in at.session_state["gt_params"] if not p)
    mask_empties = sum(1 for m in at.session_state["gt_masks"] if not m.any())
    assert empties == mask_empties, "phantom tunnels detected in GT masks"
    print(f"[ok] negative lines consistent: {empties} lines without tunnels")

    # Multi-twin: default scene has 2 tunnels -> expect per-tunnel twins,
    # per-GT-tunnel evaluation entries, and aggregated report metrics.
    twins = at.session_state["twin"]
    assert isinstance(twins, list), "twin should be a list of per-tunnel twins"
    assert len(twins) >= 1, "no twin tracks reconstructed"
    for tw in twins:
        for key in ("x_fn", "z_fn", "ys", "xs", "zs", "res_x", "res_z",
                    "v_mean", "n_det", "y0", "y1"):
            assert key in tw, f"twin missing key {key}"
        assert len(tw["res_x"]) == tw["n_det"], "residuals misaligned"
    results = at.session_state["twin_errors"]
    assert isinstance(results, list) and len(results) == 2, \
        f"expected one evaluation entry per GT tunnel, got {results}"
    n_matched = sum(1 for e in results if e["twin"])
    print(f"[ok] multi-twin: {len(twins)} tracks, {n_matched}/2 GT tunnels matched, "
          f"coverages: {[round(e['coverage'], 2) for e in results]}")

    metrics = at.session_state["report_metrics"]
    assert "tunnels_matched" in metrics and "mean_coverage" in metrics, \
        f"report missing twin aggregation: {metrics}"
    print("[ok] report metrics:", {k: round(v, 3) if isinstance(v, float) else v
                                   for k, v in metrics.items()})

    # Branded PDF report: must produce valid PDF bytes.
    from app.components import build_pdf_report
    pdf = build_pdf_report(
        at.session_state["report_metrics"],
        dict(width=10.0, length=10.0, spacing=0.5, pattern="lawnmower",
             n_tunnels=2, seed=42, device="cpu"),
        heatmap=at.session_state["heatmap"],
        extent=at.session_state["heat_extent"],
    )
    assert pdf[:5] == b"%PDF-", "PDF report is not a valid PDF"
    assert len(pdf) > 10_000, "PDF report suspiciously small"
    print(f"[ok] PDF report builds ({len(pdf) / 1024:.0f} KB)")

    # Presentation mode: toggle on (GT hidden), reveal, toggle off.
    toggles = list(at.toggle) if hasattr(at, 'toggle') else []
    pres = next((t for t in toggles if "Presentation mode" in t.label), None)
    assert pres is not None, "presentation toggle not found"
    pres.set_value(True).run()
    assert not at.exception, f"presentation mode raised: {at.exception}"
    reveal = next(b for b in at.button if "Reveal ground truth" in b.label)
    reveal.click().run()
    assert not at.exception, f"GT reveal raised: {at.exception}"
    assert at.session_state["gt_revealed"] is True
    pres = next(t for t in at.toggle if "Presentation mode" in t.label)
    pres.set_value(False).run()
    assert not at.exception, f"leaving presentation mode raised: {at.exception}"
    print("[ok] presentation mode: hide -> reveal -> off")

    # Simulate a config change -> results must be invalidated.
    # (width is capped at the model line length, 10.24 m)
    w = next(w for w in at.number_input if w.label == "Survey width [m]")
    assert abs(w.max - 10.24) < 1e-6, \
        f"width cap should be the model line length 10.24, got {w.max}"
    print("[ok] survey width capped at model line length (10.24 m)")
    w.set_value(8.0).run()
    assert not at.exception, f"App raised after config change: {at.exception}"
    assert "raw_scans" not in at.session_state, "stale results not invalidated"
    print("[ok] config change invalidates stale results")

    # 5 tunnels + small survey: defaults and previously stored widget
    # values must clamp to bounds instead of raising
    # StreamlitValueAboveMaxError.
    nt = next(w for w in at.number_input if w.label == "Number of tunnels")
    nt.set_value(5).run()
    assert not at.exception, f"App raised with 5 tunnels: {at.exception}"
    w = next(w for w in at.number_input if w.label == "Survey width [m]")
    w.set_value(4.0).run()
    assert not at.exception, \
        f"App raised after shrinking width with 5 tunnels: {at.exception}"
    print("[ok] 5 tunnels + shrunk survey: widget values clamped, no crash")

    # Random scenario: on_click callback must update tunnel widgets
    # without StreamlitAPIException.
    before_x0 = at.session_state["t0x0"] if "t0x0" in at.session_state else None
    before_y0 = at.session_state["t0y0"] if "t0y0" in at.session_state else None
    btn = next(b for b in at.button if "Random scenario" in b.label)
    btn.click().run()
    assert not at.exception, f"Random scenario raised: {at.exception}"
    assert at.session_state["t0x0"] != before_x0 or \
        at.session_state["t0y0"] != before_y0, \
        "random scenario did not update tunnel widgets"
    print("[ok] random scenario updates tunnel widgets without exception")

    # ---- French version ----
    lang = next(s for s in at.selectbox if "Language" in s.label)
    lang.set_value("Français").run()
    assert not at.exception, f"App raised in French mode: {at.exception}"
    tab_labels = [t.label for t in at.tabs]
    assert "3. Prétraitement" in tab_labels, f"Tabs not translated: {tab_labels}"
    assert any("Rapport" in t for t in tab_labels), f"Tabs not translated: {tab_labels}"
    print("[ok] French UI renders, tabs:", tab_labels)

    # Theory explanation panels must be translated too.
    markdowns = [m.value for m in at.markdown]
    assert any("Étape 1" in m for m in markdowns), \
        "Explanation panels not translated to French"
    assert any("Étape 8" in m for m in markdowns), \
        "Explanation panels not translated to French"
    print("[ok] French explanation panels render")

    # Full pipeline in French.
    btn = next(b for b in at.button if "Exécuter tout le pipeline" in b.label)
    btn.click().run()
    assert not at.exception, f"French pipeline run raised: {at.exception}"
    missing = [k for k in required if k not in at.session_state]
    assert not missing, f"Missing keys after French run: {missing}"
    print("[ok] full pipeline works in French")

    # Back to English.
    lang = next(s for s in at.selectbox if "Language" in s.label)
    lang.set_value("English").run()
    assert not at.exception, f"App raised switching back to English: {at.exception}"
    assert any("Preprocessing" in t.label for t in at.tabs), \
        [t.label for t in at.tabs]
    print("[ok] language switch back to English works")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()

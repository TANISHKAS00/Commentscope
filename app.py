"""Streamlit interface for three independently trained Jigsaw comment classifiers."""
from pathlib import Path
from html import escape
import math

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from model_runtime import CORE_FILES, LABELS, MODEL_FILES, MODEL_NAMES, ModelStore

st.set_page_config(
    page_title="CommentScope | Model review",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#eaf0f7; --muted:#9aa9bc; --line:#29384c; --teal:#59ddc4; }
    .stApp { background: radial-gradient(ellipse at 8% 0%, #16273a 0, #0b1220 40rem); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1320px; padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { border-right: 1px solid #253348; }
    [data-testid="stSidebar"] > div:first-child { background: #101a29; }
    .eyebrow { color: var(--teal); text-transform: uppercase; letter-spacing: .16em;
               font-size: .72rem; font-weight: 750; margin-bottom: .55rem; }
    .hero { font-size: clamp(2.1rem, 4vw, 3.4rem); line-height: 1.04;
            letter-spacing: -.055em; font-weight: 780; color: var(--ink); margin: 0; }
    .hero span { color: var(--teal); }
    .lede { max-width: 760px; color: var(--muted); font-size: 1.03rem;
            line-height: 1.65; margin: .8rem 0 1.35rem; }
    .section-kicker { color: var(--teal); font-size: .73rem; font-weight: 750;
                      text-transform: uppercase; letter-spacing: .13em; }
    [data-testid="stMetric"] { background: #121e2d; border: 1px solid var(--line);
        padding: 1rem 1.1rem; border-radius: 14px; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stTabs"] button { font-weight: 650; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: var(--teal); }
    [data-testid="stTextArea"] textarea { border-radius: 12px; }
    .note { border: 1px solid #32445c; background: rgba(18,30,45,.76);
        border-radius: 12px; padding: .85rem 1rem; color: #b9c7d8;
        font-size: .9rem; line-height: 1.55; margin: .7rem 0 1.2rem; }
    .stButton > button[kind="primary"] { border-radius: 10px; font-weight: 700; }
    .stDownloadButton > button { border-radius: 10px; }
    hr { border-color: #253348; }
    </style>
    """,
    unsafe_allow_html=True,
)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts_comparison"


@st.cache_resource(show_spinner="Loading trained models…")
def load_models():
    return ModelStore(ARTIFACT_DIR)


def make_rows(texts, outputs, models, store):
    rows = []
    for idx, text in enumerate(texts):
        for name in models:
            scores = outputs[name][idx]
            flagged = [
                label for j, label in enumerate(LABELS)
                if float(scores[j]) >= store.threshold(name, label)
            ]
            row = {
                "row": idx + 1,
                "comment": text,
                "model": name,
                "flag_count": len(flagged),
                "review_flag": bool(flagged),
                "flagged_labels": ", ".join(flagged) if flagged else "None",
            }
            for j, label in enumerate(LABELS):
                row[label] = round(float(scores[j]), 4)
            rows.append(row)
    return pd.DataFrame(rows)


def runtime_model_name(name):
    lowered = str(name).lower()
    if "logistic" in lowered or "tf-idf" in lowered:
        return "Logistic Regression"
    if "bilstm" in lowered:
        return "BiLSTM + Attention"
    if "textcnn" in lowered:
        return "TextCNN"
    return str(name)


def radar_chart(frame, metric, selected_models):
    """Build a self-contained SVG radar chart from per-label test metrics."""
    colors = {
        "Logistic Regression": "#59ddc4",
        "BiLSTM + Attention": "#8ca7ff",
        "TextCNN": "#ffb66e",
    }
    width, height = 760, 520
    cx, cy, radius = 380, 248, 166
    labels = LABELS
    angles = [-math.pi / 2 + 2 * math.pi * i / len(labels) for i in range(len(labels))]

    def point(value, angle):
        return cx + radius * value * math.cos(angle), cy + radius * value * math.sin(angle)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{escape(metric)} scores by label for selected models">',
        f'<rect width="{width}" height="{height}" rx="16" fill="#101a29"/>',
    ]
    for level in (0.25, 0.5, 0.75, 1.0):
        coords = [point(level, angle) for angle in angles]
        polygon = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        parts.append(f'<polygon points="{polygon}" fill="none" stroke="#34445a" stroke-width="1"/>')
        lx, ly = point(level, angles[0])
        parts.append(
            f'<text x="{lx + 5:.1f}" y="{ly + 3:.1f}" fill="#9aa9bc" '
            f'font-size="11" font-family="sans-serif">{level:.2f}</text>'
        )
    for angle in angles:
        x, y = point(1, angle)
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#34445a"/>')

    for index, label in enumerate(labels):
        x, y = point(1.16, angles[index])
        anchor = "middle"
        if x < cx - 18:
            anchor = "end"
        elif x > cx + 18:
            anchor = "start"
        readable = label.replace("_", " ").title()
        parts.append(
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="{anchor}" '
            f'fill="#dbe5f1" font-size="13" font-family="sans-serif">{escape(readable)}</text>'
        )

    plotted = 0
    for raw_model, model_rows in frame.groupby("model", sort=False):
        model = runtime_model_name(raw_model)
        if model not in selected_models or metric not in frame.columns:
            continue
        score_by_label = model_rows.set_index("label")[metric].to_dict()
        values = [max(0.0, min(1.0, float(score_by_label[label]))) for label in labels]
        coords = [point(value, angle) for value, angle in zip(values, angles)]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        color = colors.get(model, "#59ddc4")
        parts.append(
            f'<polygon points="{points}" fill="{color}" fill-opacity="0.14" '
            f'stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>'
        )
        for label, value, (x, y) in zip(labels, values, coords):
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}" stroke="#101a29" stroke-width="1.5">'
                f'<title>{escape(model)} · {escape(label)}: {value:.3f}</title></circle>'
            )
        legend_y = 474 + plotted * 20
        parts.append(f'<line x1="220" y1="{legend_y}" x2="244" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(
            f'<text x="252" y="{legend_y + 4}" fill="#dbe5f1" font-size="13" '
            f'font-family="sans-serif">{escape(model)}</text>'
        )
        plotted += 1
    parts.append("</svg>")
    return "".join(parts)


st.markdown('<div class="eyebrow">Jigsaw · comment classification</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero">Comment<span>Scope</span></h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="lede">A focused workspace for reviewing comment-level scores across '
    'Logistic Regression, BiLSTM + Attention, and TextCNN.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="note"><strong>Use with human review.</strong> These outputs describe '
    'language in comment text. They do not identify authors or predict real-world harm. '
    'Scores are classifier outputs, not calibrated probabilities.</div>',
    unsafe_allow_html=True,
)

missing = [filename for filename in CORE_FILES if not (ARTIFACT_DIR / filename).exists()]
store = None
if missing:
    st.info("Shared model files are missing. The evaluation charts below can still show exported test results.")
else:
    try:
        store = load_models()
    except Exception as exc:
        st.error(f"Model load failed: {exc}. Exported evaluation results remain available below.")

with st.sidebar:
    st.markdown('<div class="section-kicker">Workspace settings</div>', unsafe_allow_html=True)
    st.header("Models")
    available_models = store.available_models if store is not None else list(MODEL_NAMES)
    chosen = st.multiselect("Compare models", available_models, default=available_models)
    st.caption(
        "Each model uses its validation-tuned threshold for each label. "
        "Scores are not probabilities of real-world wrongdoing."
    )
    st.divider()
    if store is not None:
        st.success(f"{len(available_models)} model{'s' if len(available_models) != 1 else ''} loaded")
        st.caption("Inference runs locally in this app. No external prediction API is used.")
        missing_models = [
            f"{name} ({filename})" for name, filename in MODEL_FILES.items()
            if name not in available_models
        ]
        if missing_models:
            st.warning("Not loaded: " + ", ".join(missing_models))
    else:
        st.warning("Inference unavailable until shared model files and at least one trained model are added.")

single_tab, bulk_tab, comparison_tab = st.tabs(
    ["✳  Comment review", "⇧  Batch analysis", "▥  Model performance"]
)

with single_tab:
    if store is None:
        st.info("Add the shared model files and at least one trained checkpoint to enable comment analysis.")
    elif not chosen:
        st.info("Choose at least one model in Workspace settings to continue.")
    else:
        st.markdown('<div class="section-kicker">Single comment</div>', unsafe_allow_html=True)
        st.subheader("Review a comment")
        st.write("Enter text to compare the selected classifiers across all six Jigsaw labels.")
        comment = st.text_area(
            "Comment text",
            height=140,
            max_chars=5000,
            placeholder="Paste comment text here…",
            label_visibility="collapsed",
        )
        if st.button("Analyze comment", type="primary", use_container_width=True):
            if not comment.strip():
                st.warning("Enter some comment text first.")
            else:
                scores = store.predict([comment], chosen)
                st.markdown('<div class="section-kicker">Results</div>', unsafe_allow_html=True)
                st.markdown("#### Label score comparison")
                st.caption("Current comment · selected model scores · scale from 0 to 1")
                score_rows = [
                    {"model": name, "label": label, "score": float(scores[name][0][index])}
                    for name in chosen
                    for index, label in enumerate(LABELS)
                ]
                components.html(
                    radar_chart(pd.DataFrame(score_rows), "score", set(chosen)),
                    height=540,
                    scrolling=False,
                )
                for name in chosen:
                    values = scores[name][0]
                    thresholds = [store.threshold(name, label) for label in LABELS]
                    flags = [label for j, label in enumerate(LABELS) if values[j] >= thresholds[j]]
                    with st.container(border=True):
                        left, right = st.columns([1, 2.5], gap="large")
                        with left:
                            st.markdown(f"#### {name}")
                            st.metric("Labels for review", len(flags))
                            st.caption("Flagged: " + (", ".join(flags) if flags else "none"))
                        with right:
                            df = pd.DataFrame(
                                {
                                    "Label": LABELS,
                                    "Score": values,
                                    "Decision threshold": thresholds,
                                    "Flagged": [label in flags for label in LABELS],
                                }
                            )
                            st.dataframe(
                                df.style.format({"Score": "{:.3f}", "Decision threshold": "{:.2f}"}),
                                hide_index=True,
                                use_container_width=True,
                            )
                            st.bar_chart(
                                pd.DataFrame({"Label": LABELS, "Score": values}).set_index("Label"),
                                horizontal=True,
                            )

with bulk_tab:
    if store is None:
        st.info("Add the trained model artifacts to enable batch analysis.")
    elif not chosen:
        st.info("Choose at least one model in Workspace settings to continue.")
    else:
        st.markdown('<div class="section-kicker">CSV workflow</div>', unsafe_allow_html=True)
        st.subheader("Analyze a batch")
        st.write("Upload a CSV with a `comment_text` column, or select another text column below.")
        upload = st.file_uploader("CSV file · up to 5 MB", type=["csv"])
        if upload is not None:
            if upload.size > 5 * 1024 * 1024:
                st.error("CSV exceeds the 5 MB limit.")
            else:
                try:
                    data = pd.read_csv(upload, nrows=1001)
                    if data.empty or len(data.columns) == 0:
                        st.warning("The CSV has no rows or columns to analyze.")
                    else:
                        columns = list(data.columns)
                        default_index = columns.index("comment_text") if "comment_text" in columns else 0
                        text_col = st.selectbox("Comment column", columns, index=default_index)
                        st.caption("Blank rows are skipped. Maximum 1,000 nonempty comments per batch.")
                        if st.button("Run batch analysis", type="primary"):
                            comments = data[text_col].fillna("").astype(str).str.strip()
                            valid = comments[comments.ne("")]
                            if valid.empty:
                                st.warning("No nonempty comments found.")
                            elif len(valid) > 1000:
                                st.warning("Keep the CSV to at most 1,000 nonempty comments.")
                            else:
                                outputs = store.predict(valid.tolist(), chosen)
                                results = make_rows(valid.tolist(), outputs, chosen, store)
                                flagged_outputs = int(results.review_flag.sum())
                                m1, m2 = st.columns(2)
                                m1.metric("Comments analyzed", len(valid))
                                m2.metric("Model outputs flagged", flagged_outputs)
                                st.dataframe(results, hide_index=True, use_container_width=True)
                                st.download_button(
                                    "Download results as CSV",
                                    results.to_csv(index=False).encode("utf-8"),
                                    file_name="comment_analysis.csv",
                                    mime="text/csv",
                                )
                except Exception as exc:
                    st.error(f"Could not process CSV: {exc}")

with comparison_tab:
    st.markdown('<div class="section-kicker">Held-out evaluation</div>', unsafe_allow_html=True)
    st.subheader("Model performance")
    summary_file = ARTIFACT_DIR / "comparison_summary.csv"
    labels_file = ARTIFACT_DIR / "comparison_per_label.csv"
    if summary_file.is_file():
        summary = pd.read_csv(summary_file)
        st.caption("Metrics exported by the training notebook; they are not calculated from app submissions.")
        st.dataframe(summary, hide_index=True, use_container_width=True)
        metric_candidates = [c for c in ("macro_f1", "micro_f1", "weighted_f1") if c in summary.columns]
        if metric_candidates and "model" in summary:
            st.bar_chart(summary.set_index("model")[metric_candidates])
    else:
        st.info("Add `comparison_summary.csv` from the notebook exports to show overall test metrics.")
    if labels_file.is_file():
        per_label = pd.read_csv(labels_file)
        if {"model", "label", "f1"}.issubset(per_label.columns):
            available_metrics = [key for key in ("f1", "precision", "recall", "average_precision") if key in per_label.columns]
            metric_label = st.selectbox(
                "Radar metric",
                available_metrics,
                index=available_metrics.index("f1") if "f1" in available_metrics else 0,
                format_func=lambda key: key.replace("_", " ").title(),
            )
            normalized_selected = {runtime_model_name(name) for name in chosen}
            selected_rows = per_label[per_label["model"].map(runtime_model_name).isin(normalized_selected)]
            if selected_rows.empty:
                st.info("Select at least one model in Workspace settings to draw the radar chart.")
            else:
                st.markdown("#### Per-label comparison")
                st.caption("Validation-tuned held-out test scores · scale from 0 to 1")
                chart = radar_chart(selected_rows, metric_label, normalized_selected)
                components.html(chart, height=540, scrolling=False)
        st.markdown("#### Per-label metrics")
        st.dataframe(per_label, hide_index=True, use_container_width=True)
    else:
        st.info("Add `comparison_per_label.csv` from the notebook exports to show label-level metrics.")

st.divider()
st.caption(
    "CommentScope · Review model scores in context. Avoid submitting identifying or sensitive personal information."
)

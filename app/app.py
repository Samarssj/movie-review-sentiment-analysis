import json
import pickle
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="ReelFeel · Movie Sentiment",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "notebooks"

LABEL_MAP = {
    0: "Negative",
    1: "Somewhat Negative",
    2: "Neutral",
    3: "Somewhat Positive",
    4: "Positive",
}
LABEL_SHORT = {0: "Negative", 1: "Slightly negative", 2: "Neutral", 3: "Slightly positive", 4: "Positive"}
LABEL_COLORS = {
    0: "#ef476f",
    1: "#f78c6b",
    2: "#b7b7c9",
    3: "#70d6c5",
    4: "#ffd166",
}
LABEL_ORDER = [LABEL_MAP[i] for i in range(5)]


@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(BASE_DIR / "classical_results.json", "r") as f:
        results = json.load(f)
    return model, vectorizer, results


@st.cache_data
def load_dataset_profile():
    train_path = DATA_DIR / "train.tsv"
    if not train_path.exists():
        return None
    try:
        train = pd.read_csv(train_path, sep="\t")
        distribution = (
            train["Sentiment"].value_counts().sort_index().rename("Reviews").reset_index()
        )
        distribution.columns = ["Class", "Reviews"]
        distribution["Sentiment"] = distribution["Class"].map(LABEL_MAP)
        return distribution
    except Exception:
        return None


def set_review_text(example):
    st.session_state["review_input"] = example


model, vectorizer, results = load_artifacts()
dataset_profile = load_dataset_profile()

# -----------------------------------------------------------------------------
# Visual system: cinema black, warm marquee gold, and film-grain texture.
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700;800&display=swap');

:root {
  --ink: #f7f4ef;
  --muted: #a5a1ad;
  --panel: rgba(27, 25, 34, 0.88);
  --panel-soft: rgba(36, 32, 44, 0.72);
  --gold: #ffd166;
  --coral: #ef476f;
  --teal: #70d6c5;
  --line: rgba(255,255,255,0.09);
}

.stApp {
  color: var(--ink);
  background:
    radial-gradient(circle at 82% 3%, rgba(239,71,111,0.18), transparent 26rem),
    radial-gradient(circle at 11% 21%, rgba(112,214,197,0.10), transparent 24rem),
    linear-gradient(135deg, #0b0a0f 0%, #14111b 48%, #0b0a0f 100%);
  font-family: 'DM Sans', sans-serif;
}

.stApp:before {
  content: '';
  position: fixed;
  inset: 0;
  pointer-events: none;
  opacity: 0.13;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.36'/%3E%3C/svg%3E");
  z-index: 0;
}

.block-container { max-width: 1280px; padding: 2.4rem 4rem 4rem; position: relative; z-index: 1; }
[data-testid='stHeader'] { background: transparent; }
[data-testid='stToolbar'] { visibility: hidden; }

h1, h2, h3, h4 { font-family: 'Playfair Display', Georgia, serif !important; letter-spacing: -0.025em; }
h1 { font-size: clamp(2.8rem, 6vw, 5.5rem) !important; line-height: 0.98 !important; }
h2 { font-size: 2.1rem !important; }
h3 { font-size: 1.4rem !important; }
p, label, .stCaption { color: var(--muted); }

.hero {
  position: relative;
  overflow: hidden;
  min-height: 365px;
  padding: 3.5rem 3.6rem;
  border: 1px solid var(--line);
  border-radius: 28px;
  background:
    linear-gradient(112deg, rgba(11,10,15,0.98) 0%, rgba(11,10,15,0.80) 53%, rgba(11,10,15,0.20) 100%),
    radial-gradient(ellipse at 78% 44%, rgba(239,71,111,0.40), transparent 32%),
    radial-gradient(ellipse at 95% 104%, rgba(255,209,102,0.32), transparent 26%),
    #17131e;
  box-shadow: 0 28px 80px rgba(0,0,0,0.34);
}
.hero:after {
  content: '';
  position: absolute;
  width: 540px;
  height: 540px;
  right: -110px;
  top: -220px;
  border: 1px solid rgba(255,209,102,0.24);
  border-radius: 50%;
  box-shadow: 0 0 0 36px rgba(255,209,102,0.035), 0 0 0 72px rgba(255,209,102,0.035), 0 0 0 108px rgba(255,209,102,0.025);
}
.hero-copy { position: relative; z-index: 2; max-width: 670px; }
.eyebrow { color: var(--gold); font-size: 0.78rem; font-weight: 700; letter-spacing: 0.22em; text-transform: uppercase; margin-bottom: 1rem; }
.hero h1 { margin: 0 0 1rem; color: #fffaf2; }
.hero h1 span { color: var(--gold); }
.hero p { max-width: 570px; font-size: 1.08rem; line-height: 1.65; }

.film-strip { display: flex; gap: 10px; margin: 1.6rem 0 0; opacity: 0.72; }
.film-strip span { width: 34px; height: 12px; border: 1px solid rgba(255,209,102,0.62); border-radius: 3px; }

.section-kicker { color: var(--gold); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; margin: 2rem 0 0.5rem; }

div[data-testid='stTextArea'] textarea {
  background: rgba(15,13,20,0.84);
  color: #fffaf2;
  border: 1px solid rgba(255,255,255,0.16);
  border-radius: 16px;
  font-size: 1.02rem;
  line-height: 1.65;
  padding: 1rem;
}
div[data-testid='stTextArea'] textarea:focus { border-color: var(--gold); box-shadow: 0 0 0 1px var(--gold); }

.stButton > button {
  border: 1px solid rgba(255,209,102,0.40);
  border-radius: 999px;
  background: rgba(255,209,102,0.08);
  color: #fff7e4;
  font-weight: 600;
  transition: all 0.2s ease;
}
.stButton > button:hover { border-color: var(--gold); background: rgba(255,209,102,0.18); color: white; transform: translateY(-1px); }
.stButton > button[kind='primary'] { border: 0; background: linear-gradient(100deg, #ef476f, #f78c6b); box-shadow: 0 10px 28px rgba(239,71,111,0.22); padding: 0.7rem 1.3rem; }
.stButton > button[kind='primary']:hover { background: linear-gradient(100deg, #ff5a7f, #ff9b7d); }

[data-baseweb='tab-list'] { gap: 2rem; border-bottom: 1px solid var(--line); }
[data-baseweb='tab'] { color: var(--muted); font-weight: 600; padding: 0.8rem 0.1rem; }
[aria-selected='true'] { color: var(--gold) !important; border-bottom-color: var(--gold) !important; }

.card, .stat-card, .result-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 1.25rem 1.4rem;
}
.stat-card { min-height: 110px; }
.stat-label { color: var(--muted); font-size: 0.76rem; letter-spacing: 0.10em; text-transform: uppercase; }
.stat-value { color: #fffaf2; font-family: 'Playfair Display', Georgia, serif; font-size: 2rem; font-weight: 700; margin-top: 0.35rem; }
.stat-note { color: var(--muted); font-size: 0.82rem; margin-top: 0.2rem; }

.result-card { background: linear-gradient(145deg, rgba(40,30,43,0.95), rgba(22,20,28,0.92)); padding: 1.6rem; }
.result-label { color: var(--muted); font-size: 0.76rem; letter-spacing: 0.14em; text-transform: uppercase; }
.result-sentiment { color: var(--gold); font-family: 'Playfair Display', Georgia, serif; font-size: 2.5rem; line-height: 1.05; margin: 0.35rem 0 0.55rem; }
.result-description { color: #ded8d2; line-height: 1.5; }
.score-pill { display: inline-block; border-radius: 999px; padding: 0.35rem 0.72rem; color: #15121a; font-size: 0.78rem; font-weight: 700; background: var(--gold); }

div[data-testid='stMetric'] { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 0.8rem 1rem; }
div[data-testid='stMetricLabel'] { color: var(--muted); }
div[data-testid='stMetricValue'] { color: #fffaf2; }

[data-testid='stExpander'] { border-color: var(--line); background: rgba(22,20,28,0.76); border-radius: 16px; }
[data-testid='stDataFrame'] { border: 1px solid var(--line); border-radius: 16px; overflow: hidden; }

.footer-note { color: #777280; text-align: center; font-size: 0.78rem; margin-top: 2.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="hero-copy">
    <div class="eyebrow">ReelFeel · Sentiment Intelligence</div>
    <h1>Read the room.<br><span>Feel the film.</span></h1>
    <p>Turn a movie phrase into a five-point emotional reading. Built on Rotten Tomatoes review language, presented like a midnight screening.</p>
    <div class="film-strip"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">The screening room</div>', unsafe_allow_html=True)
st.markdown("## A closer look at your review")

# -----------------------------------------------------------------------------
# PREDICT TAB
# -----------------------------------------------------------------------------
tab_predict, tab_compare, tab_about = st.tabs(["✦ Analyze a review", "▦ Model reel", "ⓘ About the project"])

with tab_predict:
    examples = [
        "A series of escapades demonstrating the adage that what is good for the goose is also good for the gander",
        "This movie was a complete waste of time and money.",
        "An absolutely stunning, emotionally devastating masterpiece.",
        "It was okay, nothing special but not bad either.",
        "The acting was wooden and the plot made no sense.",
    ]

    left, right = st.columns([2.2, 1], gap="large")
    with left:
        st.markdown('<div class="section-kicker">01 · Your review</div>', unsafe_allow_html=True)
        text = st.text_area(
            "Enter a movie review or phrase",
            height=166,
            placeholder="e.g. The cinematography was beautiful, but the story dragged on forever.",
            label_visibility="collapsed",
            key="review_input",
        )
        analyze = st.button("Analyze sentiment", type="primary", use_container_width=True)

    with right:
        st.markdown('<div class="section-kicker">Director’s picks</div>', unsafe_allow_html=True)
        st.markdown("**Need a scene to test?**")
        for index, example in enumerate(examples):
            st.button(
                f"Scene {index + 1:02d}",
                key=f"example_{index}",
                use_container_width=True,
                on_click=set_review_text,
                args=(example,),
            )
        st.caption("Try one of the sample lines, or write your own review above.")

    if analyze and text.strip():
        with st.spinner("Projecting your sentiment..."):
            X = vectorizer.transform([text])
            pred = int(model.predict(X)[0])
            proba = np.asarray(model.predict_proba(X)[0], dtype=float)

        confidence = float(proba[pred])
        sentiment_score = float(np.dot(proba, np.arange(5)))
        normalized_score = (sentiment_score - 2) / 2
        tone = "warmly optimistic" if normalized_score > 0.35 else "cautiously positive" if normalized_score > 0.05 else "measured and neutral" if normalized_score > -0.05 else "cautiously critical" if normalized_score > -0.35 else "strongly critical"

        st.markdown('<div class="section-kicker">02 · The verdict</div>', unsafe_allow_html=True)
        result_left, result_right = st.columns([1, 1.6], gap="large")
        with result_left:
            st.markdown(
                f"""
<div class="result-card">
  <div class="result-label">Predicted sentiment</div>
  <div class="result-sentiment">{LABEL_MAP[pred]}</div>
  <span class="score-pill">{confidence:.1%} confidence</span>
  <p class="result-description">The overall tone reads as <strong>{tone}</strong>. The classifier places this phrase at {sentiment_score:.2f} on the five-point emotional scale.</p>
</div>
""",
                unsafe_allow_html=True,
            )
            st.progress(min(confidence, 1.0), text=f"Model confidence · {confidence:.1%}")

        with result_right:
            probability_df = pd.DataFrame(
                {
                    "Sentiment": LABEL_ORDER,
                    "Probability": proba,
                    "Percent": [f"{value:.1%}" for value in proba],
                }
            )
            probability_chart = (
                alt.Chart(probability_df)
                .mark_bar(cornerRadiusEnd=7, height=27)
                .encode(
                    x=alt.X("Probability:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", grid=False)),
                    y=alt.Y("Sentiment:N", sort=LABEL_ORDER, title=None, axis=alt.Axis(labelColor="#d7d0ca", labelFontSize=12, ticks=False, domain=False)),
                    color=alt.Color("Sentiment:N", scale=alt.Scale(domain=LABEL_ORDER, range=[LABEL_COLORS[i] for i in range(5)]), legend=None),
                    tooltip=[alt.Tooltip("Sentiment:N"), alt.Tooltip("Probability:Q", format=".1%")],
                )
                .properties(height=184, title=alt.TitleParams("Probability across the emotional spectrum", color="#fffaf2", font="DM Sans", fontSize=15, anchor="start"))
                .configure_view(stroke=None)
                .configure(background="transparent")
                .configure_axis(labelFont="DM Sans", titleFont="DM Sans", labelColor="#a5a1ad", titleColor="#a5a1ad", gridColor="#ffffff12")
            )
            st.altair_chart(probability_chart, use_container_width=True)

        st.markdown('<div class="section-kicker">03 · A little context</div>', unsafe_allow_html=True)
        context_a, context_b, context_c = st.columns(3)
        with context_a:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Emotional score</div><div class="stat-value">{sentiment_score:.2f}<span style="font-size:1rem;color:#a5a1ad"> / 4</span></div><div class="stat-note">Weighted across all five classes</div></div>', unsafe_allow_html=True)
        with context_b:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Dominant class</div><div class="stat-value" style="font-size:1.55rem">{LABEL_SHORT[pred]}</div><div class="stat-note">Highest probability outcome</div></div>', unsafe_allow_html=True)
        with context_c:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Phrase length</div><div class="stat-value">{len(text.split())}<span style="font-size:1rem;color:#a5a1ad"> words</span></div><div class="stat-note">Short phrases can be nuanced</div></div>', unsafe_allow_html=True)
    elif analyze:
        st.warning("Please enter a review before starting the screening.")

# -----------------------------------------------------------------------------
# MODEL COMPARISON TAB
# -----------------------------------------------------------------------------
with tab_compare:
    st.markdown('<div class="section-kicker">The model reel</div>', unsafe_allow_html=True)
    st.markdown("## How the algorithms performed")
    st.write("Every model was evaluated on the same TF-IDF features and held-out validation split. The tuned, calibrated Linear SVM powers the live predictions.")

    results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    results_df = results_df.sort_values("accuracy", ascending=False)
    best_model = results_df.iloc[0]
    avg_accuracy = results_df["accuracy"].mean()

    metric_a, metric_b, metric_c = st.columns(3)
    with metric_a:
        st.metric("Best validation accuracy", f"{best_model['accuracy']:.1%}", f"{best_model['Model']}")
    with metric_b:
        st.metric("Average model accuracy", f"{avg_accuracy:.1%}")
    with metric_c:
        st.metric("Sentiment classes", "5", "Ordinal scale")

    accuracy_chart = (
        alt.Chart(results_df)
        .mark_bar(cornerRadiusEnd=6, height=26)
        .encode(
            x=alt.X("accuracy:Q", title=None, scale=alt.Scale(domain=[0, 0.7]), axis=alt.Axis(format="%", grid=True)),
            y=alt.Y("Model:N", sort=results_df["Model"].tolist(), title=None, axis=alt.Axis(labelColor="#ded8d2", labelLimit=180, ticks=False, domain=False)),
            color=alt.condition(
                alt.datum.Model == best_model["Model"],
                alt.value("#ffd166"),
                alt.value("#70d6c5"),
            ),
            tooltip=["Model", alt.Tooltip("accuracy:Q", format=".2%")],
        )
        .properties(height=250, title=alt.TitleParams("Validation accuracy · the final cut", color="#fffaf2", font="DM Sans", fontSize=15, anchor="start"))
        .configure_view(stroke=None)
        .configure(background="transparent")
        .configure_axis(labelFont="DM Sans", titleFont="DM Sans", labelColor="#a5a1ad", titleColor="#a5a1ad", gridColor="#ffffff12")
    )
    st.altair_chart(accuracy_chart, use_container_width=True)

    runtime_df = results_df.dropna(subset=["train_time_sec"]).copy()
    if not runtime_df.empty:
        runtime_chart = (
            alt.Chart(runtime_df)
            .mark_bar(cornerRadiusEnd=6, height=22)
            .encode(
                x=alt.X("train_time_sec:Q", title=None, scale=alt.Scale(type="log"), axis=alt.Axis(title="Training time · log scale", grid=True)),
                y=alt.Y("Model:N", sort="-x", title=None, axis=alt.Axis(labelColor="#ded8d2", labelLimit=180, ticks=False, domain=False)),
                color=alt.value("#ef476f"),
                tooltip=["Model", alt.Tooltip("train_time_sec:Q", format=".2f", title="Seconds")],
            )
            .properties(height=220, title=alt.TitleParams("Training time · speed behind the scenes", color="#fffaf2", font="DM Sans", fontSize=15, anchor="start"))
            .configure_view(stroke=None)
            .configure_axis(labelFont="DM Sans", titleFont="DM Sans", labelColor="#a5a1ad", titleColor="#a5a1ad", gridColor="#ffffff12")
        )
        st.altair_chart(runtime_chart, use_container_width=True)

    if dataset_profile is not None:
        st.markdown('<div class="section-kicker">The source material</div>', unsafe_allow_html=True)
        profile_left, profile_right = st.columns([1, 1.7], gap="large")
        with profile_left:
            st.markdown('<div class="card"><div class="stat-label">Training corpus</div><div class="stat-value">{:,.0f}</div><div class="stat-note">labeled Rotten Tomatoes phrases</div></div>'.format(dataset_profile["Reviews"].sum()), unsafe_allow_html=True)
        with profile_right:
            distribution_chart = (
                alt.Chart(dataset_profile)
                .mark_arc(innerRadius=58, stroke="#14111b", strokeWidth=2)
                .encode(
                    theta=alt.Theta("Reviews:Q"),
                    color=alt.Color("Sentiment:N", scale=alt.Scale(domain=LABEL_ORDER, range=[LABEL_COLORS[i] for i in range(5)]), legend=alt.Legend(orient="right", labelColor="#a5a1ad", title=None)),
                    tooltip=["Sentiment", alt.Tooltip("Reviews:Q", format=",")],
                )
                .properties(height=230, title=alt.TitleParams("What the training set sounds like", color="#fffaf2", font="DM Sans", fontSize=15, anchor="start"))
                .configure_view(stroke=None)
                .configure(background="transparent")
            )
            st.altair_chart(distribution_chart, use_container_width=True)

    with st.expander("View the underlying validation table"):
        st.dataframe(
            results_df.style.format({"accuracy": "{:.4f}", "train_time_sec": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

# -----------------------------------------------------------------------------
# ABOUT TAB
# -----------------------------------------------------------------------------
with tab_about:
    st.markdown('<div class="section-kicker">Behind the curtain</div>', unsafe_allow_html=True)
    st.markdown("## What powers ReelFeel")
    about_left, about_right = st.columns([1.3, 1], gap="large")
    with about_left:
        st.markdown(
            """
<div class="card">
<p>This app demonstrates a <strong>five-class sentiment classifier</strong> trained on the Kaggle “Sentiment Analysis on Movie Reviews” dataset: phrase-level excerpts from Rotten Tomatoes reviews.</p>
<p><strong>Pipeline</strong></p>
<p>TF-IDF vectorization uses unigrams and bigrams across 20,000 features. Several classical algorithms are compared, with a tuned Linear SVM selected as the strongest performer. The live model is calibrated so it can return class probabilities instead of a hard label alone.</p>
<p><strong>Read the result with nuance</strong></p>
<p>The model was trained on short phrases rather than complete reviews. It can confuse adjacent classes such as neutral and somewhat positive, especially when a phrase is ironic, context-dependent, or emotionally mixed.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with about_right:
        st.markdown(
            """
<div class="card">
<div class="stat-label">Sentiment scale</div>
<div style="margin-top:1rem;line-height:2.1">
<div><span style="color:#ef476f">●</span> Negative</div>
<div><span style="color:#f78c6b">●</span> Somewhat Negative</div>
<div><span style="color:#b7b7c9">●</span> Neutral</div>
<div><span style="color:#70d6c5">●</span> Somewhat Positive</div>
<div><span style="color:#ffd166">●</span> Positive</div>
</div>
</div>
""",
            unsafe_allow_html=True,
        )

st.markdown('<div class="footer-note">A small screening room for language, tone, and the stories we tell about stories.</div>', unsafe_allow_html=True)


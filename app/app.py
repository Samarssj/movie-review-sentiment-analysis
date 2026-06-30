import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="🎬", layout="centered")

LABEL_MAP = {0: "Negative", 1: "Somewhat Negative", 2: "Neutral", 3: "Somewhat Positive", 4: "Positive"}
LABEL_COLORS = {0: "#d73027", 1: "#fc8d59", 2: "#999999", 3: "#91cf60", 4: "#1a9850"}


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(BASE_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open(BASE_DIR / "classical_results.json", "r") as f:
        results = json.load(f)

    return model, vectorizer, results


model, vectorizer, results = load_artifacts()

st.title("🎬 Movie Review Sentiment Analyzer")
st.caption(
    "Trained on the Kaggle *Sentiment Analysis on Movie Reviews* dataset "
    "(Rotten Tomatoes phrases) — 5-class sentiment, TF-IDF + Linear SVM (best classical model)."
)

tab_predict, tab_compare, tab_about = st.tabs(["🔮 Predict", "📊 Model Comparison", "ℹ️ About"])

# ---------------------------------------------------------------- PREDICT TAB
with tab_predict:
    st.subheader("Try it out")

    examples = [
        "A series of escapades demonstrating the adage that what is good for the goose is also good for the gander",
        "This movie was a complete waste of time and money.",
        "An absolutely stunning, emotionally devastating masterpiece.",
        "It was okay, nothing special but not bad either.",
        "The acting was wooden and the plot made no sense.",
    ]

    col_a, col_b = st.columns([3, 1])
    with col_b:
        st.write("**Try an example:**")
        for i, ex in enumerate(examples):
            if st.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True):
                st.session_state["review_text"] = ex

    with col_a:
        text = st.text_area(
            "Enter a movie review or phrase:",
            value=st.session_state.get("review_text", ""),
            height=140,
            placeholder="e.g. 'The cinematography was beautiful but the story dragged on forever.'",
            key="review_text",
        )

    predict_clicked = st.button("Analyze Sentiment", type="primary")

    if predict_clicked and text.strip():
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Predicted Sentiment", LABEL_MAP[pred])
            st.markdown(
                f"<div style='background-color:{LABEL_COLORS[pred]};padding:14px;"
                f"border-radius:8px;text-align:center;color:white;font-weight:bold;font-size:18px;'>"
                f"{LABEL_MAP[pred]}</div>",
                unsafe_allow_html=True,
            )
        with col2:
            proba_df = pd.DataFrame({
                "Sentiment": [LABEL_MAP[i] for i in range(5)],
                "Probability": proba,
            })
            chart = (
                alt.Chart(proba_df)
                .mark_bar()
                .encode(
                    x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("Sentiment:N", sort=None),
                    color=alt.Color(
                        "Sentiment:N",
                        scale=alt.Scale(
                            domain=[LABEL_MAP[i] for i in range(5)],
                            range=[LABEL_COLORS[i] for i in range(5)],
                        ),
                        legend=None,
                    ),
                    tooltip=["Sentiment", alt.Tooltip("Probability:Q", format=".2%")],
                )
                .properties(height=200)
            )
            st.altair_chart(chart, use_container_width=True)

        confidence = proba[pred]
        st.caption(f"Model confidence: {confidence:.1%}")
    elif predict_clicked:
        st.warning("Please enter some text first.")

# ------------------------------------------------------------- COMPARE TAB
with tab_compare:
    st.subheader("Classical ML Model Comparison")
    st.write(
        "Validation accuracy for each algorithm, trained on identical TF-IDF features "
        "(20K features, unigrams + bigrams), evaluated on a held-out validation split."
    )

    results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    results_df = results_df.sort_values("accuracy", ascending=False)

    chart2 = (
        alt.Chart(results_df)
        .mark_bar()
        .encode(
            x=alt.X("accuracy:Q", title="Validation Accuracy", scale=alt.Scale(domain=[0, 0.7])),
            y=alt.Y("Model:N", sort="-x"),
            color=alt.Color("Model:N", legend=None),
            tooltip=["Model", alt.Tooltip("accuracy:Q", format=".4f")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart2, use_container_width=True)

    st.dataframe(
        results_df.style.format({"accuracy": "{:.4f}", "macro_f1": "{:.4f}", "weighted_f1": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "**Linear SVM** (tuned, calibrated) is used in this app's predictions because it had the best "
        "validation accuracy among the classical algorithms tested."
    )

# ------------------------------------------------------------- ABOUT TAB
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
This app demonstrates a 5-class sentiment classifier trained on the **Kaggle "Sentiment Analysis on
Movie Reviews"** dataset — phrase-level excerpts from Rotten Tomatoes reviews.

**Pipeline:**
1. TF-IDF vectorization (unigrams + bigrams, 20,000 features)
2. Several classical ML algorithms compared (Naive Bayes, Logistic Regression, Linear SVM, Random Forest)
3. Best model (Linear SVM) tuned via small grid search and calibrated (`CalibratedClassifierCV`) so it
   can output class probabilities, not just hard labels
4. Full training pipeline and analysis (EDA, confusion matrices, error analysis) available in the
   companion Jupyter notebook

**Limitations:**
- The model is trained on short Rotten Tomatoes phrases, not full reviews — performance on longer,
  more nuanced free-text reviews may differ.
- Sentiment is on an ordinal 5-point scale; the model sometimes confuses adjacent classes
  (e.g. "Neutral" vs "Somewhat Positive"), which is expected given how subjective short-phrase
  labeling is.
- Class imbalance (the "Neutral" class dominates the training data) can bias predictions toward
  the middle classes for ambiguous text.
        """
    )

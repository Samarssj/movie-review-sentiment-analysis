# Sentiment Analysis on Movie Reviews

Classical ML sentiment classification (5-class, ordinal) on the Kaggle "Sentiment Analysis on Movie
Reviews" dataset (Rotten Tomatoes phrases), plus a Streamlit UI for interactive predictions.

## Project structure

```
sentiment-analysis-movie-reviews/
├── data/
│   ├── train.tsv              # Kaggle training data (PhraseId, SentenceId, Phrase, Sentiment)
│   ├── test.tsv                # Kaggle test data (no labels)
│   ├── sampleSubmission.csv    # Kaggle submission format
│   └── submission.csv          # Generated predictions on test.tsv (Linear SVM)
├── notebooks/
│   └── Sentiment_Analysis_Movie_Reviews.ipynb   # Full EDA + model training + comparison + analysis
├── app/
│   ├── app.py                  # Streamlit UI
│   ├── model.pkl               # Trained, calibrated Linear SVM model
│   ├── vectorizer.pkl          # Fitted TF-IDF vectorizer
│   ├── classical_results.json  # Accuracy results for the comparison tab
│   └── requirements.txt        # App dependencies
└── README.md
```

## Quick start

### 1. Explore the notebook
Open `notebooks/Sentiment_Analysis_Movie_Reviews.ipynb` in VS Code (with the Jupyter extension) or
Jupyter Lab. It contains:
- EDA on class distribution and phrase length
- TF-IDF feature engineering
- Training & comparison of Naive Bayes, Logistic Regression, Linear SVM, Random Forest
- Confusion matrix / error analysis
- Hyperparameter tuning
- Final predictions written to `data/submission.csv`

Install notebook dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 2. Run the Streamlit app
```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501 — type in a review, see the predicted sentiment and confidence, plus a
model comparison tab.

<img width="1440" height="900" alt="Screenshot 2026-07-01 at 12 50 11" src="https://github.com/user-attachments/assets/1f2e8ea1-b2dc-4292-9117-b33697be612a" />

<img width="1440" height="900" alt="Screenshot 2026-07-01 at 12 50 17" src="https://github.com/user-attachments/assets/7156c811-c897-4a30-b821-f1d172a1ff42" />



## Results summary

| Model | Validation Accuracy |
|---|---|
| Naive Bayes | 0.5737 |
| Logistic Regression | 0.5949 |
| Linear SVM | 0.5998 |
| Linear SVM (tuned) | 0.6012 |
| Random Forest | 0.5109 |

**Best model:** Linear SVM (TF-IDF, unigrams+bigrams, C=0.3 after tuning), calibrated with
`CalibratedClassifierCV` in the app so it can output class probabilities


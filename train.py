import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from utils import clean_text, to_sentiment

# 1. Load data (60,000 reviews keeps training fast)
df = pd.read_csv("data/Reviews.csv", usecols=["Score", "Text"]).dropna()
df = df.sample(60000, random_state=42)
df["sentiment"] = df["Score"].apply(to_sentiment)
print(df["sentiment"].value_counts())

# Chart 1: how many reviews of each type
plt.figure(figsize=(6, 4))
sns.countplot(x="sentiment", data=df, order=["negative", "neutral", "positive"])
plt.title("Sentiment distribution")
plt.savefig("outputs/distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Clean the text
print("Cleaning text... this takes a few minutes")
df["clean"] = df["Text"].apply(clean_text)

# 3. Split into train and test, then TF-IDF
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["sentiment"], test_size=0.2,
    random_state=42, stratify=df["sentiment"])

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_t = tfidf.fit_transform(X_train)
X_test_t = tfidf.transform(X_test)

# 4. Train two models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Naive Bayes": MultinomialNB(),
}

rows, trained = [], {}
for name, model in models.items():
    model.fit(X_train_t, y_train)
    pred = model.predict(X_test_t)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="weighted")
    rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                 "Precision": p, "Recall": r, "F1": f1})
    trained[name] = model
    print(f"\n{name}\n", classification_report(y_test, pred))

    labels = ["negative", "neutral", "positive"]
    cm = confusion_matrix(y_test, pred, labels=labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"outputs/cm_{name.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    plt.close()

results = pd.DataFrame(rows).round(3)
results.to_csv("outputs/results.csv", index=False)
print(results)

# 5. Top words from Logistic Regression
lr = trained["Logistic Regression"]
words = np.array(tfidf.get_feature_names_out())
classes = list(lr.classes_)
for label in ["positive", "negative"]:
    idx = np.argsort(lr.coef_[classes.index(label)])[-15:]
    plt.figure(figsize=(6, 5))
    plt.barh(words[idx], lr.coef_[classes.index(label)][idx])
    plt.title(f"Top {label} words")
    plt.savefig(f"outputs/top_{label}_words.png", dpi=150, bbox_inches="tight")
    plt.close()

# 6. Save the best model and the vectorizer
best = results.sort_values("F1", ascending=False).iloc[0]["Model"]
joblib.dump(trained[best], "models/model.pkl")
joblib.dump(tfidf, "models/tfidf.pkl")
print("Saved best model:", best)
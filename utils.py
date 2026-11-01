import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOP = set(stopwords.words("english")) - {"not", "no", "nor"}
LEMMA = WordNetLemmatizer()

def to_sentiment(score):
    if score <= 2:
        return "negative"
    if score == 3:
        return "neutral"
    return "positive"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [LEMMA.lemmatize(w) for w in text.split() if w not in STOP]
    return " ".join(words)
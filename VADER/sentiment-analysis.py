import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


df = pd.read_csv(
    "https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv",
    nrows=1000,
)


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_words = [
        token for token in tokens if token not in stopwords.words("english")
    ]
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(token) for token in filtered_words]
    processed_text = " ".join(lemmatized_text)
    return processed_text


df["reviewText"] = df["reviewText"].apply(preprocess_text)


analyzer = SentimentIntensityAnalyzer()


def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = 1 if scores["pos"] > 0 else 0
    return sentiment


df["sentiment"] = df["reviewText"].apply(get_sentiment)

print(confusion_matrix(df["Positive"], df["sentiment"]))
print(classification_report(df["Positive"], df["sentiment"]))

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Step 1: Load the dataset
df = pd.read_csv(
    "https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv",
    nrows=1000,
)


# Step 2: Preprocess the text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_words = [
        token for token in tokens if token not in stopwords.words("english")
    ]
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(token) for token in filtered_words]
    return " ".join(lemmatized_text)


df["reviewText"] = df["reviewText"].apply(preprocess_text)

# Step 3: Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["reviewText"])

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, df["Positive"], test_size=0.05, random_state=42
)

# Step 5: Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

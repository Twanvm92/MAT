import csv
import string
from typing import List
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
stop_words = set(stopwords.words("english"))

# Activate TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT_FILE = "./datasets/maldonado-dataset.csv"
OUTPUT_FILE_BINARY = (
    "features-matrices/maldonado-dataset-features-matrices-binary-classification.csv"
)
OUTPUT_FILE_MULTICLASS = "features-matrices/maldonado-dataset-features-matrices-multiclass-classification.csv"


def clean_term(text):
    text = text.lower()
    return "".join(char for char in text if char not in string.punctuation)


def standardize(text):
    stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    nltk_tokens = nltk.word_tokenize(text)
    result = ""
    for w in nltk_tokens:
        if w not in stop_words:
            text = clean_term(w)
            if not text.isdigit():
                result = result + " " + stemmer.stem(wordnet_lemmatizer.lemmatize(text))
    return result


comments = []
projects_name = []
classifications = []
BINARY_CLASSIFICATION = True

with open(INPUT_FILE) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    # Skip header
    next(csv_reader)
    for row in csv_reader:
        if BINARY_CLASSIFICATION or row[1] != "WITHOUT_CLASSIFICATION":
            comments.append(standardize(row[2]))
            projects_name.append(row[0])
            classifications.append(row[1])

vect = TfidfVectorizer()
vects = vect.fit_transform(comments)

# Select all rows from the data set
td = pd.DataFrame(vects.todense()).iloc[:]
td.columns = vect.get_feature_names_out()
term_document_matrix = td.T
term_document_matrix.columns = [
    str(i) + " - " + classifications[i - 1] for i in range(1, len(comments) + 1)
]
if BINARY_CLASSIFICATION:
    term_document_matrix["total_count"] = term_document_matrix.astype(bool).sum(axis=1)
    term_document_matrix = term_document_matrix.sort_values(
        by="total_count", ascending=False
    )[:]
    term_document_matrix = term_document_matrix.loc[
        term_document_matrix["total_count"] >= 4
    ]
    term_document_matrix = term_document_matrix.drop(columns=["total_count"])
print(term_document_matrix.T)

if BINARY_CLASSIFICATION:
    term_document_matrix.T.to_csv(OUTPUT_FILE_BINARY)
else:
    term_document_matrix.T.to_csv(OUTPUT_FILE_MULTICLASS)

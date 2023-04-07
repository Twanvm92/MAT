import csv
import pandas as pd

# Activate TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT_FILE = "./datasets/debthunter-dataset.csv"
OUTPUT_FILE_BINARY = (
    "features-matrices/debthunter-dataset-features-matrices-binary-classification.csv"
)
OUTPUT_FILE_MULTICLASS = "features-matrices/debthunter-dataset-features-matrices-multiclass-classification.csv"

comments = []
classifications = []
BINARY_CLASSIFICATION = True

with open(INPUT_FILE) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    # Skip header
    next(csv_reader)
    for row in csv_reader:
        if BINARY_CLASSIFICATION or row[1] != "WITHOUT_CLASSIFICATION":
            comments.append(row[0])
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

import string
from typing import List
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
stop_words = set(stopwords.words("english"))


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


def convert_labels_to_binary_str(labels: List[str]) -> List[int]:
    return ["1" if label.strip() == "positive" else "0" for label in labels]


class NLPPreprocessor:
    """A class for preprocessing natural language data for vectorization.

    The NLPPreprocessor class provides methods to preprocess natural language
    data for vectorization.
    The class uses the TfidfVectorizer from scikit-learn to vectorize the data and
    the PorterStemmer and WordNetLemmatizer from the nltk library for stemming
    and lemmatization, respectively.

    Attributes:
    vect (TfidfVectorizer): A TfidfVectorizer object for vectorization.
    stemmer (PorterStemmer): A PorterStemmer object for stemming.
    wordnet_lem (WordNetLemmatizer): A WordNetLemmatizer object for lemmatization.

    Methods:
    _clean_term(text): Cleans a text by converting it to lowercase and
      removing punctuation.
    _standardize(text): Standardizes a text by tokenizing it, removing stop words,
      and applying stemming and lemmatization.
    _remove_less_freq_terms(data, threshold): Removes terms that occur less frequently
      than a given threshold from a sparse matrix.
    vectorize_data(data, scen): Preprocesses a list of texts and vectorizes
      them using TfidfVectorizer.


    Example usage:
    nlp_preprocessor = NLPPreprocessor()
    preprocessed_data = nlp_preprocessor.vectorize_data(data, "train")
    """

    def __init__(self, min_df_threshold=4):
        """Initializes the NLPPreprocessor class.

        Args:
            min_df_threshold (int): The minimum document frequency threshold for
                removing terms from the vocabulary. Default is 4.
        """

        self.vect = TfidfVectorizer(min_df=min_df_threshold)
        self.stemmer = PorterStemmer()
        self.wordnet_lem = WordNetLemmatizer()

    def _clean_term(self, text):
        """Cleans a text by converting it to lowercase and removing punctuation.

        Args:
            text (str): The text to be cleaned.

        Returns:
            The cleaned text.
        """
        text = text.lower()
        return "".join(char for char in text if char not in string.punctuation)

    def _standardize(self, text):
        """Standardizes a text by tokenizing it, removing stop words, and applying
        stemming and lemmatization.

        Args:
            text (str): The text to be standardized.

        Returns:
            The standardized text.
        """
        nltk_tokens = nltk.word_tokenize(text)
        result = ""
        for w in nltk_tokens:
            if w not in stop_words:
                text = self._clean_term(w)
                if not text.isdigit():
                    result = (
                        result
                        + " "
                        + self.stemmer.stem(self.wordnet_lem.lemmatize(text))
                    )
        return result

    def vectorize_data(self, data: List, scen: str = "train") -> pd.DataFrame:
        """Preprocesses a list of texts and vectorizes them using TfidfVectorizer.

        Args:
            data (List): The list of texts to be vectorized.
            scen (str): The scenario for vectorization. Can be either "train" or
                "test". Default is "train".

        Returns:
            The vectorized data.
        """
        if scen not in ["train", "test"]:
            raise ValueError("Invalid scenario used for vectorization.")

        # add strip to remove remaining newlines
        data = [self._standardize(text.strip()) for text in data]

        if scen == "train":
            vects = self.vect.fit_transform(data)
        elif scen == "test":
            vects = self.vect.transform(data)

        return pd.DataFrame(vects.todense()).iloc[:]

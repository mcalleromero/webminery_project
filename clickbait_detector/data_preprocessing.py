from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import spacy

class Preprocessing:
    """Preprocessing class whose purpose is to cleanse, filter and extract the news' titles features.
    """

    def count_words(self, title):
        """This method counts the number of words of the title.

        Args:
            title ([str]): The news title

        Returns:
            [int]: The number of words the title is composed of
        """
        return len(title.split(" "))

    def has_question(self, title):
        """Boolean method that extracts if the news is a question or contains a question

        Args:
            title ([str]): The news title

        Returns:
            [bool]: 1 if the news title contains a question, 0 if not
        """
        title = title.lower()
        if "?" in title or title.startswith(('who','what','where','why','when','whose','whom','would','will','how','which','should','could','did','do', 'all')):
            return 1
        else:
            return 0

    def has_exclamation(self, title):
        """Boolean method that extracts if the news contains an exclamation

        Args:
            title ([str]): The news title

        Returns:
            [bool]: 1 if the news title contains an exclamation, 0 if not
        """
        if "!" in title:
            return 1
        else:
            return 0

    def starts_with_num(self, title):
        """This method purpose is to check if the news starts with a number.

        Args:
            title ([str]): The news title

        Returns:
            [bool]: 1 if the news title starts with a num, 0 if not
        """
        if title.startswith(('1','2','3','4','5','6','7','8','9')):
            return 1
        else:
            return 0

    def contains_num(self, title):
        """This method purpose is to check if the news contains a number, as clickbait news often contains numbers.

        Args:
            title ([str]): The news title

        Returns:
            [bool]: 1 if the news title contains a num, 0 if not
        """
        if any(num in title for num in ['1','2','3','4','5','6','7','8','9']):
            return 1
        else:
            return 0

    def has_parenthesis(self, title):
        """This method purpose is to check if the news contains parenthesis.

        Args:
            title ([str]): The news title

        Returns:
            [bool]: 1 if the news title contains parenthesis, 0 if not
        """
        if "(" in title or ")" in title:
            return 1
        else:
            return 0

    def num_stop_words(self, title):
        """This method counts the number of stopwords of the title using the nltk python library.

        Args:
            title ([str]): The news title

        Returns:
            [int]: The number of stopwords the title is composed of
        """
        text_tokens = word_tokenize(title)
        return sum([(word in stopwords.words()) for word in text_tokens])

    def fit_transform(self, title):
        """This function basically transforms a news title to a pandas dataframe that contains all the extracted features

        Args:
            title ([str]): The news title

        Returns:
            [dataframe]: A pandas DataFrame where each column is one of the extracted features from the news title.
        """
        nwords = self.count_words(title)
        question = self.has_question(title)
        exclamation = self.has_exclamation(title)
        starts_num = self.starts_with_num(title)
        contains_num = self.contains_num(title)
        parenthesis = self.has_parenthesis(title)
        num_stop_words = self.num_stop_words(title)

        return pd.DataFrame({'nword': nwords, 'question': question, 'exclamation': exclamation, 'starts_num': starts_num, 'contains_num': contains_num, 'parenthesis': parenthesis, 'num_stop_wrods': num_stop_words,}, index=[0])

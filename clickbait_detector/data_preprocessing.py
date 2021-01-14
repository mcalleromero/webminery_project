from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import spacy

class Preprocessing:

    def count_words(self, title):
        return len(title.split(" "))

    def has_question(self, title):
        title = title.lower()
        if "?" in title or title.startswith(('who','what','where','why','when','whose','whom','would','will','how','which','should','could','did','do', 'all')):
            return 1
        else:
            return 0

    def has_exclamation(self, title):
        if "!" in title:
            return 1
        else:
            return 0

    def starts_with_num(self, title):
        if title.startswith(('1','2','3','4','5','6','7','8','9')):
            return 1
        else:
            return 0

    def contains_num(self, title):
        if any(num in title for num in ['1','2','3','4','5','6','7','8','9']):
            return 1
        else:
            return 0

    def has_parenthesis(self, title):
        if "(" in title or ")" in title:
            return 1
        else:
            return 0

    def num_stop_words(self, title):
        text_tokens = word_tokenize(title)
        return sum([(word in stopwords.words()) for word in text_tokens])

    def fit_transform(self, title):
        nwords = self.count_words(title)
        question = self.has_question(title)
        exclamation = self.has_exclamation(title)
        starts_num = self.starts_with_num(title)
        contains_num = self.contains_num(title)
        parenthesis = self.has_parenthesis(title)
        num_stop_words = self.num_stop_words(title)

        return pd.DataFrame({'nword': nwords, 'question': question, 'exclamation': exclamation, 'starts_num': starts_num, 'contains_num': contains_num, 'parenthesis': parenthesis, 'num_stop_wrods': num_stop_words,}, index=[0])

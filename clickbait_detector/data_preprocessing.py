from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

class Preprocessing:

    def count_words(self, title):
        return len(title.split(" "))

    def has_question(self, title):
        if "?" in title or title.startswith(('who','what','where','why','when','whose','whom','would','will','how','which','should','could','did','do')):
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

    def clean_text(self, title, tokenization=False):
        title = title.lower()
        if tokenization:
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(title)
            title = ' '.join([token.lemma_ for token in doc if token.lemma_.isalpha()])
        else:
            title = title.replace('.', '').replace(',', '').replace(';', '').replace(':', '').replace("'", '').replace('"', '').replace('-', '').replace('/', '').replace('*', '').replace('+', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('’', '').replace('!', '').replace('?', '').replace('=', '')

        stopWords = set(stopwords.words("english"))
        text_tokens = word_tokenize(title)
        title = ' '.join([word for word in text_tokens if word not in stopwords.words()])

        return title


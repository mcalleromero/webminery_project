import time
from datetime import date

from guardian_crawler import GuardianSpider
from scrapy.crawler import CrawlerProcess
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize


def evaluate():
    scores = []

    f = open("../data/guardian_news_07-01-2021.txt", "r",  encoding='utf-8')
    for lines in f:
        line = lines.split("\t")
        title = line[0]
        text = line[1]
        summary = summarize(text)

    f.close()

def create_frequency_table(text_string):
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    return freqTable

def score_sentences(sentences, freqTable):
    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] // word_count_in_sentence

    return sentenceValue

def find_average_score(sentenceValue):
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    average = int(sumValues / len(sentenceValue))

    return average

def generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary

def summarize(text):
    freqTable = create_frequency_table(text)
    sentences = sent_tokenize(text)
    sentenceValue = score_sentences(sentences, freqTable)
    avgScore = find_average_score(sentenceValue)
    summary = generate_summary(sentences, sentenceValue, 6)

    return summary

if __name__ == "__main__":
    evaluate()

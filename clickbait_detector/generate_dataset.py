import os
import glob
from datetime import date
from pathlib import Path
import pandas as pd

from data_preprocessing import Preprocessing

DATA = Path('../data')

if __name__ == "__main__":
    today = date.today().strftime("%d-%m-%Y")
    filename = f'title_dataset_{today}.csv'
    final_file_path = DATA / filename

    preprocess = Preprocessing()

    news = {}
    contador = 0
    for file in DATA.glob('*.txt'):
        print(file)
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                title = line.split(':\t')[0]

                nwords = preprocess.count_words(title)
                question = preprocess.has_question(title)
                exclamation = preprocess.has_exclamation(title)
                starts_num = preprocess.starts_with_num(title)
                contains_num = preprocess.contains_num(title)
                parenthesis = preprocess.has_parenthesis(title)
                num_stop_words = preprocess.num_stop_words(title)
                #clean_title = preprocess.clean_text(title, tokenization=True)

                label = 1 #Clickbait
                if os.path.basename(file.name).split('_')[0] == "guardian":
                    label = 0 #Veridic

                news[contador] = [nwords, question, exclamation, starts_num, contains_num, parenthesis, num_stop_words, label]
                contador+=1

    pd.DataFrame.from_dict(news, orient='index', columns=['nword', 'question', 'exclamation', 'starts_num', 'contains_num', 'parenthesis', 'num_stop_words', 'label' ]).to_csv(final_file_path, index=False)

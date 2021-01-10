import os
import glob
from datetime import date
from pathlib import Path
import pandas as pd

from data_preprocessing import Preprocessing

DATA = Path('data')

if __name__ == "__main__":
    today = date.today().strftime("%d-%m-%Y")
    filename = f'title_dataset_{today}.csv'
    final_file_path = DATA / Path(filename)

    preprocess = Preprocessing()

    news = {}

    for file in Path('data').glob('*.txt'):
        with open(file, 'w+') as f:
            for line in f:
                title = line.split(':\t')[0]

                nwords = preprocess.count_words(title)
                question = preprocess.has_question(title)
                exclamation = preprocess.has_exclamation(title)
                starts_num = preprocess.starts_with_num(title)
                contains_num = preprocess.contains_num(title)
                parenthesis = preprocess.has_parenthesis(title)
                clean_title = preprocess.clean_text(title, tokenization=True)

                label = 'clickbait'
                if os.path.basename(file.name).split('_')[0] is "guardian":
                    label = 'veridic'

                news[clean_title] = [nwords, question, exclamation, starts_num, contains_num, parenthesis, label]

    pd.DataFrame(news).to_csv(final_file_path, index=False)

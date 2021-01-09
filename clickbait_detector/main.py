

from data_preprocessing import Preprocessing


if __name__ == "__main__":
    title = 'One woman’s quest to unite the diverse Hispanic community is also helping battling COVID-19!'
    p = Preprocessing()
    print(title)
    print(f"Count_words: {p.count_words(title)}")
    print(f"Question: {p.has_question(title)}")
    print(f"Exclamation: {p.has_exclamation(title)}")
    print(f"Starts with num: {p.starts_with_num(title)}")
    print(f"Contains num: {p.contains_num(title)}")
    print(f"Parenthesis: {p.has_parenthesis(title)}")
    print(f"Clean: {p.clean_text(title, tokenization=True)}")

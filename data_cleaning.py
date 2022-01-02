import pandas as pd
import re
from nltk.tokenize import word_tokenize
from dict_cleaner import clean
from nltk.stem import WordNetLemmatizer



## -- Tokenize (2) NLP Processing with Spacy -- ##
# nlp = spacy.load("en_core_web_sm", disable = ["parser", "ner"])
# df['doc'] = df['comment_text'].apply(nlp)

# df_val['less_toxic_doc'] = df_val['less_toxic'].apply(nlp)
# df_val['more_toxic_doc'] = df_val['more_toxic'].apply(nlp)

# df_test['test_doc'] = df_test['text'].apply(nlp)


## -- Tokenize (1) -- ##
removed_punc = {",", ".", "\n", "-", "*", "\"", "\'", "\`", "(", ")", "=", ">", "<"}
link_pat = re.compile(r"(Talk|User|Wikipedia|File|MediaWiki|Template|Help|Category|Portal|Draft|TimedText|Module|wp)(_talk)?:[^\s\"']+", flags=re.IGNORECASE)
sig_date_pat = re.compile(r"\d+:\d+, [A-Za-z]+ \d+, \d\d\d\d \(UTC\)")
http_pat = re.compile(r"http(s)?://[^\s\"]+")

lemmatizer = WordNetLemmatizer()
def data_processing(row):
    row = re.sub(link_pat, "[INTERNAL_LINK]", row)
    row = re.sub(sig_date_pat, "[SIG_DATE]", row)
    row = re.sub(http_pat, "[LINK]", row)
    row = clean(row)
    a = word_tokenize(row)
    # ls = [str(n) for n in a]
    ls = [lemmatizer.lemmatize(n) for n in a if n not in removed_punc]
    return ls


def generate_clean_data():
    ## -- Load Data -- ##
    df = pd.read_csv("Data/train.csv")
    df_val = pd.read_csv("Data/validation_data.csv")
    df_test = pd.read_csv("Data/comments_to_score.csv", index_col = False)
    
    df['doc'] = df['comment_text'].apply(data_processing)
    df_val['less_toxic_doc'] = df_val['less_toxic'].apply(data_processing)
    df_val['more_toxic_doc'] = df_val['more_toxic'].apply(data_processing)
    df_test['test_doc'] = df_test['text'].apply(data_processing)

    df.to_pickle('train.pickle')
    df_val.to_pickle('val.pickle')
    df_test.to_pickle('test.pickle')

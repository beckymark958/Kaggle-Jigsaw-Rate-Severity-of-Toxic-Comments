## -- Import -- ##
from nltk import data
import pandas as pd
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import re
from nltk.stem import WordNetLemmatizer

## -- Reload Saved Data -- ##
# df = pd.read_pickle("processed_spacy.pickle")
# df_val = pd.read_pickle("processed_spacy_vali.pickle")
# df_test = pd.read_pickle("processed_spacy_test.pickle")

## -- Load Data -- ##
df = pd.read_csv("Data/train.csv")
df_val = pd.read_csv("Data/validation_data.csv")
df_test = pd.read_csv("Data/comments_to_score.csv", index_col = False)

# print(df.shape)
# print(df.head) # columns: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate


## -- Create toxicity level -- ##
# df[df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) != 0].max()
df['toxic'] = df['toxic'] * 0.32
df['severe_toxic'] = df['severe_toxic'] * 1.5
df['obscene'] = df['obscene'] * 0.16
df['threat'] = df['threat'] * 1.5
df['insult'] = df['insult'] * 0.64
df['identity_hate'] = df['identity_hate'] * 1.5

# df['severe_toxic'] = df['severe_toxic'] * 2
df['level'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis = 1)
level_set = df['level'].unique()
level_set.sort()
new_levels = {}
for i in range(len(level_set)):
    new_levels[level_set[i]] = i / len(level_set)
df['level'] = df['level'].map(new_levels)

## -- Sampling Data -- ##
df_zero = df[df['level'] == 0].sample(10000, random_state = 0)
df_nonzero = df[df['level'] != 0]
df = pd.concat([df_zero, df_nonzero])


plt.hist(df['level'], bins = np.arange(0, 5, 0.5))
plt.show()


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

## -- Additional Features -- ##

# Raw Text's Punctuation Percentage
punc_list = {".", "!", "?", "*", "&", "$", "@", '#'}
def count_punctuation(tokenized_row):
    count = 0
    for char in tokenized_row:
        if char in punc_list:
            count += 1
    return count/len(tokenized_row)

# Raw Text's Uppercase Percentage
def count_upper(row):
    return sum(1 for c in row if c.isupper())

# Raw Text's Average Sentence Length
def avg_sentence_len(row):
    s_list = row.split(".")
    sum = 0
    for s in s_list:
        sum += len(s)
    return sum/len(s_list)

# Raw Text's Profanity Word Count
profane_words = set()
with open("Data/profane-words.txt", "r") as f:
    for line in f.readlines():
        profane_words.add(lemmatizer.lemmatize(line.strip("\n")))

def count_profane_words(tokenized_row):
    tokenized_row = list(map(str.lower, tokenized_row))
    count = 0
    for word in tokenized_row:
        if word in profane_words:
            count += 1
    return count/len(tokenized_row)

def word_length(row):
    return np.mean(list(map(len, row)))


## -- Additional Features Selection -- ##
def add_features(tokenized, untokenized, *, punc = False, upper = False, sentence_len = False, profane = False, length = False):
    return_vals = []
    if punc:
        return_vals.append(untokenized.apply(count_punctuation))
    if upper:
        return_vals.append(untokenized.apply(count_upper))
    if sentence_len:
        return_vals.append(untokenized.apply(avg_sentence_len))
    if profane:
        return_vals.append(tokenized.apply(count_profane_words))
    if length:
        return_vals.append(tokenized.apply(word_length))
    return return_vals   

df['upper_%'], df['punc_%'] = add_features(None, df['comment_text'], upper = True, punc = True)
df_val['less_upper_%'], df_val['less_punc_%'] = add_features(None, df_val['less_toxic'], upper = True, punc = True)
df_val['more_upper_%'], df_val['more_punc_%'] = add_features(None, df_val['more_toxic'], upper = True, punc = True)
df_test['upper_%'], df_test['punc_%'] = add_features(None, df_test['text'], upper = True, punc = True)


## -- Tokenize (2) NLP Processing with Spacy -- ##
# nlp = spacy.load("en_core_web_sm", disable = ["parser", "ner"])
# df['doc'] = df['comment_text'].apply(nlp)

# df_val['less_toxic_doc'] = df_val['less_toxic'].apply(nlp)
# df_val['more_toxic_doc'] = df_val['more_toxic'].apply(nlp)

# df_test['test_doc'] = df_test['text'].apply(nlp)


## -- Save processed data -- ##
# df.to_pickle("processed_lemma.pickle")
# df_val.to_pickle("processed_lemma_vali.pickle")
# df_test.to_pickle("processed_lemma_test.pickle")



## -- Data Cleaning -- ##
def clean(col):
    
    # data[col] = data[col].str.replace('https?://\S+|www\.\S+', ' social medium ')      
        
    col = col.lower()
    col = col.replace("4", "a")
    col = col.replace("2", "l")
    col = col.replace("5", "s")
    col = col.replace("1", "i")
    col = col.replace("!", "i")
    col = col.replace("|", "i")
    col = col.replace("0", "o")
    col = col.replace("l3", "b")
    col = col.replace("7", "t")
    col = col.replace("7", "+")
    col = col.replace("8", "ate")
    col = col.replace("3", "e")
    col = col.replace("9", "g")
    col = col.replace("6", "g")
    col = col.replace("@", "a")
    col = col.replace("$", "s")
    col = col.replace("#ofc", " of fuckin course ")
    col = col.replace("fggt", " faggot ")
    col = col.replace("your", " your ")
    col = col.replace("self", " self ")
    col = col.replace("cuntbag", " cunt bag ")
    col = col.replace("fartchina", " fart china ")
    col = col.replace("youi", " you i ")
    col = col.replace("cunti", " cunt i ")
    col = col.replace("sucki", " suck i ")
    col = col.replace("pagedelete", " page delete ")
    col = col.replace("cuntsi", " cuntsi ")
    col = col.replace("i'm", " i am ")
    col = col.replace("offuck", " of fuck ")
    col = col.replace("centraliststupid", " central ist stupid ")
    col = col.replace("hitleri", " hitler i ")
    col = col.replace("i've", " i have ")
    col = col.replace("i'll", " sick ")
    col = col.replace("fuck", " fuck ")
    col = col.replace("f u c k", " fuck ")
    col = col.replace("shit", " shit ")
    col = col.replace("bunksteve", " bunk steve ")
    col = col.replace('wikipedia', ' social medium ')
    col = col.replace("faggot", " faggot ")
    col = col.replace("delanoy", " delanoy ")
    col = col.replace("jewish", " jewish ")
    col = col.replace("sexsex", " sex ")
    col = col.replace("allii", " all ii ")
    col = col.replace("i'd", " i had ")
    col = col.replace("'s", " is ")
    col = col.replace("youbollocks", " you bollocks ")
    col = col.replace("dick", " dick ")
    col = col.replace("cuntsi", " cuntsi ")
    col = col.replace("mothjer", " mother ")
    col = col.replace("cuntfranks", " cunt ")
    col = col.replace("ullmann", " jewish ")
    col = col.replace("mr.", " mister ")
    col = col.replace("aidsaids", " aids ")
    col = col.replace("njgw", " nigger ")
    col = col.replace("wiki", " social medium ")
    col = col.replace("administrator", " admin ")
    col = col.replace("gamaliel", " jewish ")
    col = col.replace("rvv", " vanadalism ")
    col = col.replace("admins", " admin ")
    col = col.replace("pensnsnniensnsn", " penis ")
    col = col.replace("pneis", " penis ")
    col = col.replace("pennnis", " penis ")
    col = col.replace("pov.", " point of view ")
    col = col.replace("vandalising", " vandalism ")
    col = col.replace("cock", " dick ")
    col = col.replace("asshole", " asshole ")
    col = col.replace("youi", " you ")
    col = col.replace("afd", " all fucking day ")
    col = col.replace("sockpuppets", " sockpuppetry ")
    col = col.replace("iiprick", " iprick ")
    col = col.replace("penisi", " penis ")
    col = col.replace("warrior", " warrior ")
    col = col.replace("loil", " laughing out insanely loud ")
    col = col.replace("vandalise", " vanadalism ")
    col = col.replace("helli", " helli ")
    col = col.replace("lunchablesi", " lunchablesi ")
    col = col.replace("special", " special ")
    col = col.replace("ilol", " i lol ")
    col = col.replace(r'\b[uU]\b', 'you')
    col = col.replace(r"what's", "what is ")
    col = col.replace(r"\'s", " is ")
    col = col.replace(r"\'ve", " have ")
    col = col.replace(r"can't", "cannot ")
    col = col.replace(r"n't", " not ")
    col = col.replace(r"i'm", "i am ")
    col = col.replace(r"\'re", " are ")
    col = col.replace(r"\'d", " would ")
    col = col.replace(r"\'ll", " will ")
    col = col.replace(r"\'scuse", " excuse ")
    col = col.replace('\s+', ' ')  # will remove more than one whitespace character
    col = col.replace(r'(.)\1+', r'\1\1') # 2 or more characters are replaced by 2 characters
    col = col.replace("[:|♣|'|§|♠|*|/|?|=|%|&|-|#|•|~|^|>|<|►|_]", '')
    
    col = col.replace(r"what's", "what is ")
    col = col.replace(r"\'ve", " have ")
    col = col.replace(r"can't", "cannot ")
    col = col.replace(r"n't", " not ")
    col = col.replace(r"i'm", "i am ")
    col = col.replace(r"\'re", " are ")
    col = col.replace(r"\'d", " would ")
    col = col.replace(r"\'ll", " will ")
    col = col.replace(r"\'scuse", " excuse ")
    col = col.replace(r"\'s", " ")

    # Clean some punctutations
    col = col.replace('\n', ' \n ')
    col = col.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)',r'\1 \2 \3')
    # Replace repeating characters more than 3 times to length of 3
    col = col.replace(r'([*!?\'])\1\1{2,}',r'\1\1\1')    
    # Add space around repeating characters
    col = col.replace(r'([*!?\']+)',r' \1 ')
    # patterns with repeating characters 
    col = col.replace(r'([a-zA-Z])\1{2,}\b',r'\1\1')
    col = col.replace(r'([a-zA-Z])\1\1{2,}\B',r'\1\1\1')
    col = col.replace(r'[ ]{2,}',' ').strip()
    col = col.replace(r'[ ]{2,}',' ').strip()
    return col

# df["comment_text"] = clean(df, "comment_text")
# df_val['less_toxic']  = clean(df_val, "less_toxic")
# df_val['more_toxic'] = clean(df_val, "more_toxic")
# df_test['text'] = clean(df_test, "text")


df['doc'] = df['comment_text'].apply(data_processing)
df_val['less_toxic_doc'] = df_val['less_toxic'].apply(data_processing)
df_val['more_toxic_doc'] = df_val['more_toxic'].apply(data_processing)
df_test['test_doc'] = df_test['text'].apply(data_processing)


df['profane_%'], df['word_length'] = add_features(df['doc'], df['comment_text'], profane = True, length=True)

df_val['less_profane_%'], df_val['less_word_length'] = add_features(df_val['less_toxic_doc'], df_val['less_toxic'], profane = True, length=True)
df_val['more_profane_%'], df_val['more_word_length'] = add_features(df_val['more_toxic_doc'], df_val['more_toxic'], profane = True, length=True)

df_test['profane_%'], df_test['word_length'] = add_features(df_test['test_doc'], df_test['text'], profane = True, length=True)


## -- Feature Visualization -- ##
# df[df['length'].max() == df['length']].comment_text.iloc[0]
# plt.hist(df['upper_%'])
# plt.show()
# feature_df = df[['length', 'punc_%', 'upper_%', 'sentence_len', 'level']]
# feature_df.head()
# feature_df.corr()


## -- Create X -- ##
def create_vectors(tokenized_train_doc, df_val, df_test, text_only, *, min_df = 150):
    # tfidf_vectorizer = TfidfVectorizer(max_df = 0.7, min_df = 150, stop_words = "english")
    
    def nltktokenizer(text):
        if type(text) is list:
            return [t.lower() for t in text]
        return text
        # return[t for t in text]
    tfidf_vectorizer = TfidfVectorizer(max_df = 0.8, min_df = min_df, tokenizer = nltktokenizer, lowercase = False, stop_words = "english")

    train = tfidf_vectorizer.fit_transform(tokenized_train_doc)#.apply(lambda x: " ".join(x)))
    more = tfidf_vectorizer.transform(df_val['more_toxic_doc'])#.apply(lambda x: " ".join(x).lower()))
    less = tfidf_vectorizer.transform(df_val['less_toxic_doc'])#.apply(lambda x: " ".join(x).lower()))
    test = tfidf_vectorizer.transform(df_test['test_doc'])#.apply(lambda x: " ".join(x)))

    if text_only:
        return (train, more, less, test)
    else:
        
        train_X = []
        for vector, prof, punc, upper, word_len in zip(train, df['profane_%'], df['punc_%'], df['upper_%'], df['word_length']):
            train_X.append(np.append(vector.toarray(), [prof, punc, upper, word_len]))

        more_X = []
        for vector, prof, punc, upper, word_len in zip(more, df_val['more_profane_%'], df_val['more_punc_%'], df_val['more_upper_%'], df_val['more_word_length']):
            more_X.append(np.append(vector.toarray(), [prof, punc, upper, word_len]))

        less_X = []
        for vector, prof, punc, upper, word_len in zip(less, df_val['less_profane_%'], df_val['less_punc_%'], df_val['less_upper_%'], df_val['less_word_length']):
            less_X.append(np.append(vector.toarray(), [prof, punc, upper, word_len]))

        test_X = []
        for vector, prof, punc, upper, word_len in zip(test, df_test['profane_%'], df_test['punc_%'], df_test['upper_%'], df_test['word_length']):
            test_X.append(np.append(vector.toarray(), [prof, punc, upper, word_len]))
    
        return (train_X, more_X, less_X, test_X)

train_X, more_X, less_X, test_X = create_vectors(df['doc'], df_val, df_test, text_only = False, min_df = 30)


## -- Training & Validating & Testing -- ##
def train_val_test(model, train_X, more_X, less_X, test_X, df_test = None):
    model.fit(train_X, df['level'])

    more_scores = model.predict(more_X)
    less_scores = model.predict(less_X)

    validation_scores = (more_scores - less_scores) > 0
    print(str(model) + " validation score:", sum(validation_scores) / len(validation_scores))

    if df_test is not None:
        output = df_test.copy()
        output['score'] = model.predict(test_X)
        # output[['comment_id', 'score']].to_csv('test_output.csv', index = None)


## -- Linear Regression -- ##
# Validation Best: 0.6760993755812409 (with weight)
reg = LinearRegression()
train_val_test(reg, train_X, more_X, less_X, test_X, df_test)

## -- Random Forest Regression -- ##
# Validation Best: 0.6679287896904477
# rf = RandomForestRegressor(n_jobs = -1)
# train_val_test(rf, train_X, more_X, less_X, test_X, df_test)

## -- Support Vector Regression -- ##
# Validation Best: 0.6711837385412515
# svr = SVR()
# train_val_test(svr, train_X, more_X, less_X, test_X, df_test)

## -- Linear SVR -- ##
lsvr = LinearSVR(random_state = 530896, max_iter=6000)
train_val_test(lsvr, train_X, more_X, less_X, test_X, df_test)

# 530896
# 0.6787564766839378

## -- Ridge Regression -- ##
# Validation Best: 0.670087684336389 (min_df = 30, alpha = 5, with lemma & clean)
# rg = Ridge(alpha = 5)
# train_val_test(rg, train_X, more_X, less_X, test_X, df_test)

## -- Lasso Regression -- ##
# lasso = linear_model.Lasso(alpha = 0.1)
# train_val_test(lasso, train_X, more_X, less_X, test_X, df_test)

## -- KNN -- ##
# Validation Best: 0.6262123023781054 (k = 10, prof, punc, upper)
# knn = KNeighborsRegressor(n_neighbors = 10, n_jobs=-1)
# train_val_test(knn, train_X, more_X, less_X, test_X, df_test)


## -- Feature Importance -- ##
# importances = rg.feature_importances_
# indicies = np.argsort(-importances)[0:10]
# plt.barh(range(10), importances[indicies])

# plt.yticks(range(10), labels = np.array(tfidf_vectorizer.get_feature_names_out())[indicies])
# plt.show()
# np.array(tfidf_vectorizer.get_feature_names())[indicies]


    
from nltk.stem import WordNetLemmatizer
import numpy as np

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


lemmatizer = WordNetLemmatizer()
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


    
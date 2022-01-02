from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

## -- Create X -- ##
def create_vectors(tokenized_train_doc, df, df_val, df_test, text_only, *, min_df = 150):
    # tfidf_vectorizer = TfidfVectorizer(max_df = 0.7, min_df = 150, stop_words = "english")
    
    def nltktokenizer(text):
        if type(text) is list:
            return [t.lower() for t in text]
        return text
        # return[t for t in text]
    tfidf_vectorizer = TfidfVectorizer(max_df = 0.8, min_df = min_df, tokenizer = nltktokenizer, lowercase = False, stop_words = "english")

    train = tfidf_vectorizer.fit_transform(tokenized_train_doc)
    more = tfidf_vectorizer.transform(df_val['more_toxic_doc'])
    less = tfidf_vectorizer.transform(df_val['less_toxic_doc'])
    test = tfidf_vectorizer.transform(df_test['test_doc'])

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


## -- Training & Validating & Testing -- ##
def train_val_test(model, y, train_X, more_X, less_X, test_X, df_test = None):
    model.fit(train_X, y)

    more_scores = model.predict(more_X)
    less_scores = model.predict(less_X)

    validation_scores = (more_scores - less_scores) > 0
    print(str(model) + " validation score:", sum(validation_scores) / len(validation_scores))

    if df_test is not None:
        output = df_test.copy()
        output['score'] = model.predict(test_X)
        # output[['comment_id', 'score']].to_csv('test_output.csv', index = None)

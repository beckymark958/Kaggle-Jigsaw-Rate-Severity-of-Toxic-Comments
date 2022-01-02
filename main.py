## -- Import -- ##
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from add_features import add_features


if __name__ == "__main__":
    ## -- Recreate cleaned data -- ##
    # from data_cleaning import generate_clean_data
    # generate_clean_data()

    ## -- Reload Saved Data -- ##
    df = pd.read_pickle("train.pickle")
    df_val = pd.read_pickle("val.pickle")
    df_test = pd.read_pickle("test.pickle")

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


    df['upper_%'], df['punc_%'] = add_features(None, df['comment_text'], upper = True, punc = True)
    df_val['less_upper_%'], df_val['less_punc_%'] = add_features(None, df_val['less_toxic'], upper = True, punc = True)
    df_val['more_upper_%'], df_val['more_punc_%'] = add_features(None, df_val['more_toxic'], upper = True, punc = True)
    df_test['upper_%'], df_test['punc_%'] = add_features(None, df_test['text'], upper = True, punc = True)


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


    from model_training_lib import create_vectors, train_val_test
    train_X, more_X, less_X, test_X = create_vectors(df['doc'], df, df_val, df_test, text_only = False, min_df = 30)


    ## -- Linear Regression -- ##
    # Validation Best: 0.6760993755812409 (with weight)
    reg = LinearRegression()
    train_val_test(reg, df['level'], train_X, more_X, less_X, test_X, df_test)

    ## -- Random Forest Regression -- ##
    # Validation Best: 0.6679287896904477
    # rf = RandomForestRegressor(n_jobs = -1)
    # train_val_test(rf, df['level'], train_X, more_X, less_X, test_X, df_test)

    ## -- Support Vector Regression -- ##
    # Validation Best: 0.6711837385412515
    # svr = SVR()
    # train_val_test(svr, df['level'], train_X, more_X, less_X, test_X, df_test)

    ## -- Linear SVR -- ##
    lsvr = LinearSVR(random_state = 530896, max_iter=6000)
    train_val_test(lsvr, df['level'], train_X, more_X, less_X, test_X, df_test)

    # 530896
    # 0.6787564766839378

    ## -- Ridge Regression -- ##
    # Validation Best: 0.670087684336389 (min_df = 30, alpha = 5, with lemma & clean)
    # rg = Ridge(alpha = 5)
    # train_val_test(rg, df['level'], train_X, more_X, less_X, test_X, df_test)

    ## -- Lasso Regression -- ##
    # lasso = linear_model.Lasso(alpha = 0.1)
    # train_val_test(lasso, df['level'], train_X, more_X, less_X, test_X, df_test)

    ## -- KNN -- ##
    # Validation Best: 0.6262123023781054 (k = 10, prof, punc, upper)
    # knn = KNeighborsRegressor(n_neighbors = 10, n_jobs=-1)
    # train_val_test(knn, df['level'], train_X, more_X, less_X, test_X, df_test)


    ## -- Feature Importance -- ##
    # importances = rg.feature_importances_
    # indicies = np.argsort(-importances)[0:10]
    # plt.barh(range(10), importances[indicies])

    # plt.yticks(range(10), labels = np.array(tfidf_vectorizer.get_feature_names_out())[indicies])
    # plt.show()
    # np.array(tfidf_vectorizer.get_feature_names())[indicies]


        
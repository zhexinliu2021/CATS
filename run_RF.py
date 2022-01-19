#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2,SelectFpr,f_classif,mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import  gridsch
import importlib
from sklearn.utils import shuffle
from scipy.ndimage.filters import gaussian_filter1d
import multiprocessing
# In[ ]:


import multiprocessing

def preprocess():
    df = pd.read_csv('train.csv')
    X = df.drop(['samples','subtypes'], axis = 1)
    Y = df.iloc[:,1]
    X = X.to_numpy(); Y = Y.to_numpy()
    #score_list = SelectKBest(score_func=mutual_info_classif, k = X.shape[1]).fit(X,Y).scores_
    #order = np.array(list(reversed(np.argsort(score_list))))
    return (X,Y)


def run_rf(iter_num, ran_state, rank_time, X, Y):
    out_file = open('output_' + rank_time + '/' + str(iter_num) + '_result', 'w')

    X_, Y_ = shuffle(X, Y)

    out_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=ran_state)
    for out_index, (train_index, test_index) in enumerate(out_cv.split(X_, Y_), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        acc_out_arry = np.array([])

        # current_list = SelectKBest(score_func=mutual_info_classif, k = X.shape[1]).fit(X_train, y_train).scores_
        # order = np.array(list(reversed(np.argsort(current_list))))
        n_feature = 150
        if rank_time == 'part':
            current_score_list = SelectKBest(score_func=mutual_info_classif, k=n_feature).fit(X_train, y_train).scores_
            current_order = np.array(list(reversed(np.argsort(current_score_list))))
            order = current_order
        elif rank_time != 'part':
            current_score_list = SelectKBest(score_func=mutual_info_classif, k=n_feature).fit(X_, Y_).scores_
            current_order = np.array(list(reversed(np.argsort(current_score_list))))
            order = current_order

        # use forward filetering to get the optimal number of reporters.
        for iteration in range(n_feature):
            feature_index = order[:iteration + 1]
            X_train_tran = X_train[:, feature_index]  # transform the feature space.

            # each feature we apply a 10 fold inner cv to get a accuracy.
            int_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=ran_state)
            acc_inner = np.array([])

            for train_i_index, test_i_index in int_cv.split(X_train, y_train):
                X_train_i, X_test_i = X_train_tran[train_i_index], X_train_tran[test_i_index]
                y_train_i, y_test_i = y_train[train_i_index], y_train[test_i_index]

                # each inner cv we build a model using the reporter assigned.
                clf = RandomForestClassifier(n_estimators=50, max_samples=0.8, bootstrap=True, n_jobs=1)
                clf.fit(X_train_i, y_train_i)
                ypre = clf.predict(X_test_i)
                acc = accuracy_score(ypre, y_test_i)
                acc_inner = np.append(acc_inner, acc)
            # when the 10X is finished, we average the acc and add to the out_arry
            acc_out_arry = np.append(acc_out_arry, np.mean(acc_inner))

            # select the features with the maximum accuracy, and build a model using all X_train
        smoothed = gridsch.smooth(acc_out_arry, sigma=4, bool_=False)
        optimal_features = [(acc, index) for (index, acc) in enumerate(smoothed) if acc == np.max(smoothed)]
        op_index = ''
        op_index = optimal_features[-1][1] if len(optimal_features) != 1 else optimal_features[0][1]
        # plt.plot(acc_out_arry)

        feature_index = order[:op_index + 1]
        X_train_tran, X_test_tran = X_train[:, feature_index], X_test[:, feature_index]

        # gridsearch here
        rf = RandomForestClassifier()
        rf_random = GridSearchCV(estimator=rf, param_grid=gridsch.random_grid, cv=3,
                                 verbose=0, n_jobs=1, refit=True)

        rf_random.fit(X_train_tran, y_train)
        # print(rf_random.best_params_)

        # use the model with the optimal paremeters.
        # best_random  = rf_random.best_estimator_
        # clf_final = RandomForestClassifier(n_estimators=100, max_samples=0.8, bootstrap= True, n_jobs=1)
        clf_final = rf_random.best_estimator_
        clf_final.fit(X_train_tran, y_train)
        # prediction on the validation set.
        ypre = clf_final.predict(X_test_tran)
        acc = accuracy_score(ypre, y_test)
        # print('acc on the validation set is: ',acc)

        out_file.write('ACC: ' + str(acc) + '\n')
        out_file.write('features: ' + ' '.join(feature_index.astype(str)) + '\n')
        out_file.write('parameters: ' + str(rf_random.best_params_)+'\n')
    out_file.close()


# In[ ]:


# parallize the program. 
def out_loop():

    X,Y = preprocess()

    # parallize the program.
    for i in range(10):
        rank_time = 'part'
        p_list = []
        for j in range(1, 11):
            index = i * 10 + j
            p = multiprocessing.Process(target=run_rf, args=(index, index, rank_time,X, Y))
            p_list.append(p)

        # start the programs.
        for p_p in p_list:
            p_p.start()

        for p_p in p_list:
            p_p.join()

        print(str(i * 10) + ' has been done. ')

    for i in range(10):
        rank_time = 'all'
        p_list = []
        for j in range(1, 11):
            index = i * 10 + j
            p = multiprocessing.Process(target=run_rf, args=(index, index, rank_time,X,Y))
            p_list.append(p)

        # start the programs.
        for p_p in p_list:
            p_p.start()

        for p_p in p_list:
            p_p.join()

        print(str(i * 10) + ' has been done. ')








if __name__ == '__main__':

    #out_loop()
    #gridsch.get_outputs('/home/lab706/jerryliu/CATS/outter_loop/output_part')
    #gridsch.get_outputs('/home/lab706/jerryliu/CATS/outter_loop/output_all')


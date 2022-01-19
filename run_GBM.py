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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
import multiprocessing
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

def preprocess():
    df = pd.read_csv('train.csv')
    X = df.drop(['samples','subtypes'], axis = 1)
    Y = df.iloc[:,1]
    X = X.to_numpy(); Y = Y.to_numpy()
    #score_list = SelectKBest(score_func=mutual_info_classif, k = X.shape[1]).fit(X,Y).scores_
    #order = np.array(list(reversed(np.argsort(score_list))))
    return (X,Y)



def run_rgb(iter_num, ran_state, rank_time,X, Y, feature_slet ):
    if feature_slet == True:

        out_file = open('output_' + rank_time + '/' + str(iter_num) + '_result', 'w')
    else:
        out_file = open('output_no_featureselt_' + rank_time + '/' + str(iter_num) + '_result', 'w')

    X_, Y_ = shuffle(X, Y)
    out_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=ran_state)

    for out_index, (train_index, test_index) in enumerate(out_cv.split(X_, Y_), start=1):
        X_train, X_test = X_[train_index], X_[test_index]
        y_train, y_test = Y_[train_index], Y_[test_index]
        acc_out_arry = np.array([])

        # current_list = SelectKBest(score_func=mutual_info_classif, k = X.shape[1]).fit(X_train, y_train).scores_
        # order = np.array(list(reversed(np.argsort(current_list))))
        n_feature = 100
        if rank_time == 'part':
            current_score_list = SelectKBest(score_func=mutual_info_classif, k=n_feature).fit(X_train, y_train).scores_
            current_order = np.array(list(reversed(np.argsort(current_score_list))))
            order = current_order
        elif rank_time != 'part':
            current_score_list = SelectKBest(score_func=mutual_info_classif, k=n_feature).fit(X_, Y_).scores_
            current_order = np.array(list(reversed(np.argsort(current_score_list))))
            order = current_order

        if feature_slet == True:

            # use forward filetering to get the optimal number of reporters.
            for iteration in (range(n_feature)):
                feature_index = order[:iteration + 1]
                X_train_tran = X_train[:, feature_index]  # transform the feature space.

                # each feature we apply a 10 fold inner cv to get a accuracy.
                int_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=ran_state)
                acc_inner = np.array([])

                for train_i_index, test_i_index in int_cv.split(X_train, y_train):
                    X_train_i, X_test_i = X_train_tran[train_i_index], X_train_tran[test_i_index]
                    y_train_i, y_test_i = y_train[train_i_index], y_train[test_i_index]

                    # each inner cv we build a model using the reporter assigned.
                    #                 scaler = MinMaxScaler()
                    #                 X_train_i = scaler.fit_transform(X_train_i)
                    #                 X_test_i = scaler.transform(X_test_i)

                    gb = GradientBoostingClassifier()
                    # gb = HistGradientBoostingClassifier(l2_regularization = 1)
                    gb.fit(X_train_i, y_train_i)
                    ypre = gb.predict(X_test_i)
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
        else:
            op_index = len(order)-1


        feature_index = order[:op_index + 1]
        X_train_tran, X_test_tran = X_train[:, feature_index], X_test[:, feature_index]

        # gridsearch with learning rate = 0.1, and n_estimator = 95.
        # scaler = MinMaxScaler()
        X_train_tran_sca = X_train_tran  # scaler.fit_transform(X_train_tran)
        X_test_tran_sca = X_test_tran  # scaler.transform(X_test_tran)

        GB = GradientBoostingClassifier(learning_rate=0.1, n_estimators=95)
        # GB = HistGradientBoostingClassifier()
        gsearch = GridSearchCV(estimator=GB, param_grid=gridsch.gridsearch_bgm, cv=3,
                               verbose=0, n_jobs=1, refit=True)

        gsearch.fit(X_train_tran_sca, y_train)
        final_GMB = gsearch.best_estimator_
        final_GMB.fit(X_train_tran_sca, y_train)

        ypre = final_GMB.predict(X_test_tran_sca)
        acc = accuracy_score(ypre, y_test)

        out_file.write('ACC: ' + str(acc) + '\n')
        out_file.write('features: ' + ' '.join(feature_index.astype(str)) + '\n')
        out_file.write('parameters: ' + str(gsearch.best_params_) + '\n')
    out_file.close()

def out_loop():

    X,Y = preprocess()
    feature_select_bool = True
    # parallize the program.
    for i in range(10):
        rank_time = 'part'
        p_list = []
        for j in range(1, 11):
            index = i * 10 + j
            p = multiprocessing.Process(target=run_rgb, args=(index, index, rank_time,X, Y, feature_select_bool))
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
            p = multiprocessing.Process(target=run_rgb, args=(index, index, rank_time,X,Y, feature_select_bool))
            p_list.append(p)

        # start the programs.
        for p_p in p_list:
            p_p.start()

        for p_p in p_list:
            p_p.join()

        print(str(i * 10) + ' has been done. ')

def out_loop_no_feature_sect():
    X, Y = preprocess()
    feature_select_bool = False
    for i in range(10):
        rank_time = 'part'
        p_list = []
        for j in range(1, 11):
            index = i * 10 + j
            p = multiprocessing.Process(target=run_rgb, args=(index, index, rank_time,X, Y, feature_select_bool))
            p_list.append(p)

        # start the programs.
        for p_p in p_list:
            p_p.start()

        for p_p in p_list:
            p_p.join()

        print(str(i * 10) + ' has been done. ')

if __name__ == '__main__':

    #out_loop_no_feature_sect()
    #gridsch.get_outputs('/home/lab706/jerryliu/CATS/outter_loop_GBM/output_part', gridsch.gridsearch_bgm)
    #gridsch.get_outputs('/home/lab706/jerryliu/CATS/outter_loop_GBM/output_all',gridsch.gridsearch_bgm)
    gridsch.get_outputs('/home/lab706/jerryliu/CATS/outter_loop_GBM/output_no_featureselt_part', gridsch.gridsearch_bgm)
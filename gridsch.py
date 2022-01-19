import os
os.environ['OMP_NUM_THREADS'] = "1"
import pandas as pd
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 3)]
#max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num =3 )]
max_depth.append(None)
min_samples_split = [2,3]
min_samples_leaf = [1, 2, 3]
bootstrap = [True]
max_samples = [0.8]

#------------   BGM --------------------------------------------
random_grid = {'n_estimators': n_estimators,

               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

max_depth_BGM = [int(x) for x in np.linspace(60, 110, num =5 )]
max_depth_BGM.append(None)

gridsearch_bgm = {
'max_depth': max_depth_BGM,
'min_samples_split': [2,3],
'min_samples_leaf': [1,2,3],
'max_features' : [None, 'sqrt']
}

#------------   XGBoost --------------------------------------------

gridsearch_xgb = {
'max_depth': [4,6,8],
'min_child_weight': [round(x,1) for x in np.linspace(0.8,1.2 ,num = 4)]  ,
"gamma" :[i/10.0 for i in range(0,2)],
"subsample":[i/10.0 for i in range(6,9)],
"colsample_bytree" : [i/10.0 for i in range(6,9)]

}

def smooth(y, sigma=4, bool_ = True):
    if bool_:

        return gaussian_filter1d(y,sigma = sigma)
    else:
        return(y)


def get_outputs(path, pra_template):
    #extact 300 accuracies.
    acc_list= []; feature_list = []; pra_list = []

    for i in range(1,101):
        file_name = str(i)+'_result'
        file = open(path+'/'+file_name, 'r')
        for line in file:
            line = line.strip()
            if line.startswith('ACC'): acc_list.append(float(line.split(':')[1].strip()))

            elif line.startswith('features'):
                f = line.split(':')[1].split()
                feature_list.append(f)

            elif line.startswith('parameters'):
                pra = eval(line.replace('parameters:',''))
                pra_list.append(pra)

        file.close()
    feature_list = ['\t'.join(list_f)   for list_f in feature_list]

    # add the parameters to the df_dict.
    df_dict = {key:[] for key in pra_template.keys()}
    for pra_dict in pra_list:
        key_list = pra_dict.keys()
        for key in key_list:
            df_dict[key].append(pra_dict[key])

    df_dict['acc'] = acc_list; df_dict['features'] = feature_list
    #df = pd.DataFrame({'acc': acc_list, 'features' : feature_list,
    #                   'max_depth':max_depth, 'min_samples_leaf':min_samples_leaf,'min_samples_split':\
    #                    min_samples_split})

    pd.DataFrame(df_dict).to_csv(path+'/All_result.csv')







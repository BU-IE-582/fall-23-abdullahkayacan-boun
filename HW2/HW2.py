# %% [markdown]
# # IE 582 - HW2 - 5 Datasets Analysis

# %% [markdown]
# Prepared by Abdullah Kayacan

# %% [markdown]
# ## Datasets

# %% [markdown]
# - All criteria needed for the dataset are met
#   - Each dataset has pre-defined training and test sets 

# %% [markdown]
# ### 1. IDA2016
# 
# - **Source**: https://archive.ics.uci.edu/dataset/414/ida2016challenge
# - **Objective**: Finding whether Air Pressure System (APS) that generates sufficient air pressure for a truck to function correctly fails or not in the given cases.
# - **Number of attributes**: 171 (Continuous: 164, Ordinal: 7)
# - **Problem Type**: Binary classification
# - Anonymized features.
# 
# ####
# - **Criteria met**:
#   - p>>100
#   - Imbalanced data (train positive rate: 0.0167, test positive rate: 0.0234)
#   - Non-continuous features
# - We handled missing data by simply taking the average for each column in the training set. The same is valid for the ordinal values.

# %% [markdown]
# ### 2. HillValey
# 
# - **Source**: https://archive.ics.uci.edu/dataset/166/hill+valley
# - **Objective**: Predictiong whether a given instance is hill or valley by evaluating the y coordinate of the sequenced points.
# - **Number of attributes**: 100 (Continuous: 100) 
# - The features are the y coordinate for the given sequence. We preferred noisy data in the assignment.
# - Here are examples for noisy ve smoothed data
# 
# ![Hill Valley Example Data](Hill_Valley_visual_examples.jpg "Hill Valley Example Data")
# 

# %% [markdown]
# ### 3. Smartphone
# 
# - **Source**: https://archive.ics.uci.edu/dataset/364/smartphone+dataset+for+human+activity+recognition+har+in+ambient+assisted+living+aal
# - **Objective**: Activity recognition by using spatial features acquired by the phones of the users.
# - **Number of attributes**: 561 (Continuous: 561) 
# - The features are the some descriptive statistics (mean, std, min, max ,iqr, entropy, corr, etc.) of the collected velocity and acceleration data.
# - **Classes**:
#   - **0**: *WALKING*
#   - **1**: *WALKING_UPSTAIRS*
#   - **2**: *WALKING_DOWNSTAIRS*
#   - **3**: *SITTING*
#   - **4**: *STANDING*
#   - **5**: *LAYING*
# 
# ####
# - **Criteria met**:
#   - p>>100
#   - Multi-class

# %% [markdown]
# ### 4. YearPrediction
# 
# - **Source**: https://archive.ics.uci.edu/dataset/203/yearpredictionmsd
# - **Objective**: Prediction of the year of a given song by its musical attributes
# - **Number of attributes**: 90 (Continuous: 90) 
# - The feautres are average (12 of them) andcovariance (78) of timbre in song.
# 
# ####
# - **Criteria met**:
#   - Regression

# %% [markdown]
# ### 5. Reuters
# 
# - **Source**: https://archive.ics.uci.edu/dataset/217/reuter+50+50
# - **Objective**: Prediction of the author of a given article content.
# - **Number of attributes**: 100 (Categorical/Binary: 100)
# - In the trainin set, words in all texts are extracted after some preprocessing. Then, stop-words are removed. 100 most used words are taken as variable in our models (1 if word is in the text, 0 otherwise).
# - **Classes**: 50 authors. Number of texts are evenly divided.
# 
# ####
# - **Criteria met**:
#   - Multi-Class
#   - Non-continuous features

# %% [markdown]
# ## Module Imports

# %%
import os
import pickle
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from lightgbm import LGBMClassifier,LGBMRegressor

from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support,mean_squared_error,mean_absolute_error

import re

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# %% [markdown]
# ## Functions Used

# %% [markdown]
# ### To get data smoothly:

# %%
def get_data(data_source):
    if data_source == "IDA2016":
        train = pd.read_csv("IDA2016Challenge/aps_failure_training_set.csv",skiprows=20)
        test = pd.read_csv("IDA2016Challenge/aps_failure_test_set.csv",skiprows=20)
    elif data_source == "HillValley":
        train = pd.read_csv("HillValley/Hill_Valley_with_noise_Training.data")
        test = pd.read_csv("HillValley/Hill_Valley_with_noise_Testing.data")
        train = pd.concat([train.iloc[:,-1],train.iloc[:,:-1]],axis=1)
        test = pd.concat([test.iloc[:,-1],test.iloc[:,:-1]],axis=1)
    elif data_source == "Smartphone":
        train_X = pd.read_csv("Smartphone/final_X_train.txt",header=None)
        train_y = pd.read_csv("Smartphone/final_y_train.txt",header=None)-1
        train = pd.concat([train_y,train_X],axis=1).T.reset_index(drop=True).T
        train.iloc[:,0] = train.iloc[:,0].astype(int)
        test_X = pd.read_csv("Smartphone/final_X_test.txt",header=None)
        test_y = pd.read_csv("Smartphone/final_y_test.txt",header=None)-1
        test = pd.concat([test_y,test_X],axis=1).T.reset_index(drop=True).T
        test.iloc[:,0] = train.iloc[:,0].astype(int)
        del train_X,train_y,test_X,test_y
    elif data_source == "YearPrediction":
        """
        Data source indicates that:
        * You should respect the following train / test split:
        * train: first 463,715 examples
        * test: last 51,630 examples
        """
        data = pd.read_csv("YearPrediction/YearPredictionMSD.txt",header=None)
        n_train = 463715
        train,test = data.iloc[:n_train],data.iloc[n_train:]
    else:
        authors = os.listdir("Reuters/C50train")
        train = []
        test = []
        for author in authors:
            train_files = [f'Reuters/C50train/{author}/{e}' for e in os.listdir(f'Reuters/C50train/{author}')]
            for train_file in train_files:
                with open(train_file, 'r') as file:
                    content = file.read()
                train.append([author,content])

            
            test_files = [f'Reuters/C50test/{author}/{e}' for e in os.listdir(f'Reuters/C50test/{author}')]
            for test_file in test_files:
                with open(test_file, 'r') as file:
                    content = file.read()
                test.append([author,content])
        train = pd.DataFrame(train)
        test = pd.DataFrame(test)

    return train,test


# %% [markdown]
# ### Dictionary flattening
# This function is taken from https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary

# %%
### This function is taken from https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
def flatten_dict(nested_dict,exceptions = []):
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                if key not in exceptions:
                    key = list(key)
                    key.insert(0, k)
                    res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res


# %% [markdown]
# ### Dictionary-to-List function

# %%

def dict_to_list(d):
    res = []
    for k,v in d.items():
        k = list(k)
        k.append(v)
        res.append(k)
    return res

# %% [markdown]
# ## Data Retrieval & Preprocessing

# %%
data_for_models = {}

train,test = get_data(data_source="IDA2016")
train = train.replace("na",np.nan)
test = test.replace("na",np.nan)

print("IDA2016")

train.isna().mean().sort_values(ascending=False).plot(title="Missing value rates in train data",xlabel="Columns",ylabel="Frequency")
plt.show()
test.isna().mean().sort_values(ascending=False).plot(title="Missing value rates in test data",xlabel="Columns",ylabel="Frequency")
plt.show()
print("---")

featıres = train.columns[1:]
target_col = train.columns[0]
train[featıres] = train[featıres].astype(float)
filler = train[featıres].mean()
train[featıres] = train[featıres].fillna(filler)
test[featıres] = test[featıres].fillna(filler)
train[target_col] = train[target_col].map({"neg":0,"pos":1})
test[target_col] = test[target_col].map({"neg":0,"pos":1})
train[target_col] = train[target_col].astype(int)
test[target_col] = test[target_col].astype(int)


data_for_models["IDA2016"] = {"train": train.copy(), "test":test.copy()}

train,test = get_data(data_source="HillValley")
data_for_models["HillValley"] = {"train": train.copy(), "test":test.copy()}

print("HillValley")
print("---")

train,test = get_data(data_source="Smartphone")
data_for_models["Smartphone"] = {"train": train.copy(), "test":test.copy()}

print("Smartphone")
print("---")

train,test = get_data(data_source="YearPrediction")
data_for_models["YearPrediction"] = {"train": train.copy(), "test":test.copy()}

print("YearPrediction")
print("---")

train,test = get_data(data_source="Reuters")
train[1] = [re.sub('[^A-Za-z0-9]+', ' ', e.lower()).replace("  "," ") for e in train[1]]
train[1] = [re.sub('\d+', ' ', e).replace("  "," ").split(" ") for e in train[1]]
test[1] = [re.sub('[^A-Za-z0-9]+', ' ', e.lower()).replace("  "," ") for e in test[1]]
test[1] = [re.sub('\d+', ' ', e).replace("  "," ").split(" ") for e in test[1]]

train_all_words = np.array([item for sublist in train[1].values for item in sublist])
train_all_words = train_all_words[~np.isin(train_all_words,stopwords.words('english'))]
word_counts = pd.DataFrame(np.unique(train_all_words,return_counts=True)).T
word_counts.sort_values(1,ascending=False).iloc[:100,0]
features = word_counts.sort_values(1,ascending=False).iloc[:100,0].values

train = pd.concat([train.iloc[:,:1],pd.DataFrame([np.isin(features,e) for e in train[1].values])],axis=1).T.reset_index(drop=True).T
test = pd.concat([test.iloc[:,:1],pd.DataFrame([np.isin(features,e) for e in test[1].values])],axis=1).T.reset_index(drop=True).T

authors = sorted(list(set(train.iloc[:,0])))
authors_to_num = {a:i for i,a in enumerate(authors)}
train = train.replace(authors_to_num)
test = test.replace(authors_to_num)

data_for_models["Reuters"] = {"train": train.copy(), "test":test.copy()}

print("Reuters")

for data_name in data_for_models.keys():
    np.random.seed(1)
    data_for_models[data_name]["skf"] = StratifiedKFold(n_splits=5)
    data_for_models[data_name]["skf"].get_n_splits(data_for_models[data_name]["train"].iloc[:,1:], data_for_models[data_name]["train"].iloc[:,0])

    data_for_models[data_name]["binary_classification"] = True if len(set(data_for_models[data_name]["train"].iloc[:,0])) == 2 else False
    data_for_models[data_name]["classification"] = False if data_name == "YearPrediction" else True

# %% [markdown]
# ## Descriptive Statistics of Datasets

# %% [markdown]
# #### IDA2016

# %%
data_name = "IDA2016"

# %% [markdown]
# **Descriptive Statistics**:

# %%
data_for_models[data_name]["train"].describe()

# %% [markdown]
# **Correlation Heatmap**:

# %%
sns.heatmap(data_for_models[data_name]["train"].corr())

# %% [markdown]
# #### HillValley

# %%
data_name = "HillValley"

# %% [markdown]
# **Descriptive Statistics**:

# %%
data_for_models[data_name]["train"].describe()

# %% [markdown]
# **Correlation Heatmap**:

# %%
sns.heatmap(data_for_models[data_name]["train"].corr())

# %% [markdown]
# #### Smartphone

# %%
data_name = "Smartphone"

# %% [markdown]
# **Class Counts**:

# %%
data_for_models[data_name]["train"].iloc[:,0].value_counts()

# %% [markdown]
# **Descriptive Statistics**:

# %%
data_for_models[data_name]["train"].describe()

# %% [markdown]
# **Correlation Heatmap**:

# %%
sns.heatmap(data_for_models[data_name]["train"].corr())

# %% [markdown]
# #### YearPrediction

# %%
data_name = "YearPrediction"

# %% [markdown]
# **Output Distribution**:

# %%
data_for_models[data_name]["train"].iloc[:,0].hist()
plt.title("Output: Release year of the songs")

# %% [markdown]
# **Descriptive Statistics**:

# %%
data_for_models[data_name]["train"].describe()

# %% [markdown]
# **Correlation Heatmap**:

# %%
sns.heatmap(data_for_models[data_name]["train"].corr())

# %% [markdown]
# #### Reuters

# %%
data_name = "Reuters"

# %% [markdown]
# **Class Counts**:

# %%
pd.DataFrame(data_for_models[data_name]["train"].iloc[:,0].value_counts()).sort_index().T

# %% [markdown]
# **Descriptive Statistics**:

# %%
data_for_models[data_name]["train"].astype(int).describe()

# %% [markdown]
# **Correlation Heatmap**:

# %%
sns.heatmap(data_for_models[data_name]["train"].astype(int).corr())

# %% [markdown]
# ### Cross-Validation Runs

# %% [markdown]
# - 5-Fold CV was applied.
# - Algorithms and tuned parameters:
#     - **KNN** (*knn*):
#       - **distance measure** (*p*): takes 1 for L1, 2 for L2 norm
#       - **number of neigbors** (*k*): 1 to 5
#     - **Decision Trees** (*dt*)
#       - **minimal number of observations per tree leaf** (*min_samples_leaf*): 1 to 5 since it is suggested in this range.
#       - **minimum number of observations to split**: twice of *min_samples_leaf*
#     - **Random Forest** (*rf*)
#       - **number of features** (*m*): takes *1*, *sqrt(p)/2*, *sqrt(p)*, and *sqrt(p)*2*
#     - **LightGBM** (*gbm*)
#       - **max depth** (*md*): takes 1,3,7, and -1 (no limit)
#       - **number of trees** (*nt*): takes 100, 250, and 500
#       - **learning rate** (*lr*): takes 0.05, 0.1, and 0.2
# - Calculated performance metrics
#   - For binary classification problems: confusion matrix, roc-auc, precision, recall, f-score
#   - For mult-class classification problems: confusion matrix, precision, recall, f-score
#   - For regression problem: mean squared error, mean absolute error

# %%
cv_results = {d:{m:{} for m in ["knn","dt","rf","gbm"]} for d in data_for_models.keys()}
cv_results = joblib.load("cv_results.pkl")
joblib.dump(cv_results,"cv_results.pkl")
it = 0
for data_name in data_for_models.keys():
    skf = data_for_models[data_name]["skf"]
    train = data_for_models[data_name]["train"]
    test = data_for_models[data_name]["test"]

    for i, (train_index, test_index) in enumerate(skf.split(train.iloc[:,1:], train.iloc[:,0])):
        # train_index = train_index[:100]
        # test_index = test_index[:100]
        y_train,X_train = train.iloc[train_index,0],train.iloc[train_index,1:]
        y_valid,X_valid = train.iloc[test_index,0],train.iloc[test_index,1:]

        model_name = "knn"
        distance_metrics = [1,2]  # Minkowski metric parameter, 1 for manhattan, 2 for euclidean
        n_neighbors = [1,2,3,4,5]  # Number of neighbours
        

        cv_results[data_name][model_name][i] = {}
        for p in distance_metrics:
            cv_results[data_name][model_name][i][p] = {}
            for k in n_neighbors:
                cv_results[data_name][model_name][i][p][k] = {}

                model_function = KNeighborsClassifier if data_for_models[data_name]["classification"] else KNeighborsRegressor
                model = model_function(n_jobs=-1,n_neighbors=k,p=p)
                model.fit(X_train, y_train)

                if data_for_models[data_name]["classification"]:
                    pred_proba = model.predict_proba(X_valid)
                    pred = pred_proba.argmax(1)
                    cv_results[data_name][model_name][i][p][k]["cm"] = confusion_matrix(y_valid, pred)
                    if data_for_models[data_name]["binary_classification"]:
                        cv_results[data_name][model_name][i][p][k]["ra"] = roc_auc_score(y_valid, pred_proba[:,1])
                    cv_results[data_name][model_name][i][p][k]["prfs"] = precision_recall_fscore_support(y_valid, pred)
                    it += 1
                else:
                    pred = model.predict(X_valid)
                    cv_results[data_name][model_name][i][p][k]["mse"] = mean_squared_error(y_valid, pred)
                    cv_results[data_name][model_name][i][p][k]["mae"] = mean_absolute_error(y_valid, pred)
                # break
            # break
        joblib.dump(cv_results,"cv_results.pkl")

        model_name = "dt"
        min_samples_leaf = [1,2,3,4,5]

        cv_results[data_name][model_name][i] = {}
        for m in min_samples_leaf:
            cv_results[data_name][model_name][i][m] = {}
            model_function = DecisionTreeClassifier if data_for_models[data_name]["classification"] else DecisionTreeRegressor
            model = model_function(random_state=0,ccp_alpha=0,min_samples_leaf=m,min_samples_split=m*2)
            model.fit(X_train, y_train)

            if data_for_models[data_name]["classification"]:
                pred_proba = model.predict_proba(X_valid)
                pred = pred_proba.argmax(1)
                cv_results[data_name][model_name][i][m]["cm"] = confusion_matrix(y_valid, pred)
                if data_for_models[data_name]["binary_classification"]:
                    cv_results[data_name][model_name][i][m]["ra"] = roc_auc_score(y_valid, pred_proba[:,1])
                cv_results[data_name][model_name][i][m]["prfs"] = precision_recall_fscore_support(y_valid, pred)
            else:
                pred = model.predict(X_valid)
                cv_results[data_name][model_name][i][m]["mse"] = mean_squared_error(y_valid, pred)
                cv_results[data_name][model_name][i][m]["mae"] = mean_absolute_error(y_valid, pred)
            it += 1
            # break
        joblib.dump(cv_results,"cv_results.pkl")

        model_name = "rf"
        n_features = train.shape[1]-1
        max_features = ["1","sqrt/2","sqrt","sqrt*2"]
        max_feature_values = [1,np.rint(np.sqrt(n_features)/2).astype(int),np.rint(np.sqrt(n_features)).astype(int),np.rint(np.sqrt(n_features)*2).astype(int)]
        max_features_map = {max_features[m]:max_feature_values[m] for m in range(len(max_features))}

        cv_results[data_name][model_name][i] = {}
        for m in max_features:
            cv_results[data_name][model_name][i][m] = {}
            model_function = RandomForestClassifier if data_for_models[data_name]["classification"] else RandomForestRegressor
            model = model_function(random_state=0,n_jobs=3,max_features=max_features_map[m])
            model.fit(X_train, y_train)

            if data_for_models[data_name]["classification"]:
                pred_proba = model.predict_proba(X_valid)
                pred = pred_proba.argmax(1)
                cv_results[data_name][model_name][i][m]["cm"] = confusion_matrix(y_valid, pred)
                if data_for_models[data_name]["binary_classification"]:
                    cv_results[data_name][model_name][i][m]["ra"] = roc_auc_score(y_valid, pred_proba[:,1])
                cv_results[data_name][model_name][i][m]["prfs"] = precision_recall_fscore_support(y_valid, pred)
            else:
                pred = model.predict(X_valid)
                cv_results[data_name][model_name][i][m]["mse"] = mean_squared_error(y_valid, pred)
                cv_results[data_name][model_name][i][m]["mae"] = mean_absolute_error(y_valid, pred)
            it += 1
            # break
        joblib.dump(cv_results,"cv_results.pkl")

        model_name = "gbm"
        n_features = train.shape[1]-1
        max_depths = [1,3,7,-1]
        n_trees = [100,250,500]
        learning_rates = [.05,.1,.2]

        cv_results[data_name][model_name][i] = {}
        for md in max_depths:
            cv_results[data_name][model_name][i][md] = {}
            for nt in n_trees:
                cv_results[data_name][model_name][i][md][nt] = {}
                for lr in learning_rates:
                    cv_results[data_name][model_name][i][md][nt][lr] = {}
                    model_function = LGBMClassifier if data_for_models[data_name]["classification"] else LGBMRegressor
                    model = model_function(random_state=0,n_jobs=-1,verbose=-1,max_depth=md,n_estimators=nt,learning_rate=lr)
                    model.fit(X_train, y_train)

                    if data_for_models[data_name]["classification"]:
                        pred_proba = model.predict_proba(X_valid)
                        pred = pred_proba.argmax(1)
                        cv_results[data_name][model_name][i][md][nt][lr]["cm"] = confusion_matrix(y_valid, pred)
                        if data_for_models[data_name]["binary_classification"]:
                            cv_results[data_name][model_name][i][md][nt][lr]["ra"] = roc_auc_score(y_valid, pred_proba[:,1])
                        cv_results[data_name][model_name][i][md][nt][lr]["prfs"] = precision_recall_fscore_support(y_valid, pred)
                    else:
                        pred = model.predict(X_valid)
                        cv_results[data_name][model_name][i][md][nt][lr]["mse"] = mean_squared_error(y_valid, pred)
                        cv_results[data_name][model_name][i][md][nt][lr]["mae"] = mean_absolute_error(y_valid, pred)
                        it += 1
                    # break
            #     break
            # break
        joblib.dump(cv_results,"cv_results.pkl")



# %% [markdown]
# ### CV Evaluation

# %% [markdown]
# - Average of 5-folds for each algorithm in each dataset is used to compare
# - Metrics used to decide test parameter selection:
#   - ROC-AUC is used for **IDA2016** dataset due to the class imbalance problem.
#   - Accuracy metric is calculated for the remaining classification problems.
#   - For the regression problem, mean squared error is preferred since it is more responsive for the high individual errors.
# - The code below outputs the top three algorithm parameters for the algorithms and datasets. The parameters in the first row of each table is used in test models.

# %%
cv_model_results_to_compare = {data_name:[] for data_name in data_for_models.keys()}
data_name = "IDA2016"
print(data_name)
# df_model_perfs
print("knn")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["knn"])))
df_model = df_model[df_model.iloc[:,3] == "ra"].groupby([1,2])[4].mean().reset_index().sort_values(4,ascending=False)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("dt")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["dt"])))
df_model = df_model[df_model.iloc[:,2] == "ra"].groupby([1])[3].mean().reset_index().sort_values(3,ascending=False)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("rf")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["rf"])))
df_model = df_model[df_model.iloc[:,2] == "ra"].groupby([1])[3].mean().reset_index().sort_values(3,ascending=False)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("gbm")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["gbm"])))
df_model = df_model[df_model.iloc[:,4] == "ra"].groupby([1,2,3])[5].mean().reset_index().sort_values(5,ascending=False)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("----------------------------------")
print("----------------------------------")

data_name = "HillValley"
print(data_name)

print("knn")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["knn"])))
df_model = df_model[df_model.iloc[:,3] == "cm"]
df_model[4] = [e.diagonal().sum()/e.sum() for e in df_model[4]]
df_model = df_model.groupby([1,2])[4].mean().reset_index().sort_values(4,ascending=False)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("dt")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["dt"])))
df_model = df_model[df_model.iloc[:,2] == "cm"]
df_model[3] = [e.diagonal().sum()/e.sum() for e in df_model[3]]
df_model = df_model.groupby([1])[3].mean().reset_index().sort_values(3,ascending=False)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("rf")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["rf"])))
df_model = df_model[df_model.iloc[:,2] == "cm"]
df_model[3] = [e.diagonal().sum()/e.sum() for e in df_model[3]]
df_model = df_model.groupby([1])[3].mean().reset_index().sort_values(3,ascending=False)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("gbm")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["gbm"])))
df_model = df_model[df_model.iloc[:,4] == "cm"]
df_model[5] = [e.diagonal().sum()/e.sum() for e in df_model[5]]
df_model = df_model.groupby([1,2,3])[5].mean().reset_index().sort_values(5,ascending=False)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("----------------------------------")
print("----------------------------------")

data_name = "Smartphone"
print(data_name)

print("knn")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["knn"])))
df_model = df_model[df_model.iloc[:,3] == "cm"]
df_model[4] = [e.diagonal().sum()/e.sum() for e in df_model[4]]
df_model = df_model.groupby([1,2])[4].mean().reset_index().sort_values(4,ascending=False)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("dt")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["dt"])))
df_model = df_model[df_model.iloc[:,2] == "cm"]
df_model[3] = [e.diagonal().sum()/e.sum() for e in df_model[3]]
df_model = df_model.groupby([1])[3].mean().reset_index().sort_values(3,ascending=False)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("rf")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["rf"])))
df_model = df_model[df_model.iloc[:,2] == "cm"]
df_model[3] = [e.diagonal().sum()/e.sum() for e in df_model[3]]
df_model = df_model.groupby([1])[3].mean().reset_index().sort_values(3,ascending=False)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("gbm")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["gbm"])))
df_model = df_model[df_model.iloc[:,4] == "cm"]
df_model[5] = [e.diagonal().sum()/e.sum() for e in df_model[5]]
df_model = df_model.groupby([1,2,3])[5].mean().reset_index().sort_values(5,ascending=False)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("----------------------------------")
print("----------------------------------")

data_name = "YearPrediction"
print(data_name)

print("knn")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["knn"])))
df_model = df_model[df_model.iloc[:,3] == "mse"].groupby([1,2])[4].mean().reset_index().sort_values(4,ascending=True)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("dt")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["dt"])))
df_model = df_model[df_model.iloc[:,2] == "mse"].groupby([1])[3].mean().reset_index().sort_values(3,ascending=True)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("rf")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["rf"])))
df_model = df_model[df_model.iloc[:,2] == "mse"].groupby([1])[3].mean().reset_index().sort_values(3,ascending=True)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("gbm")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["gbm"])))
df_model = df_model[df_model.iloc[:,4] == "mse"].groupby([1,2,3])[5].mean().reset_index().sort_values(5,ascending=True)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("----------------------------------")
print("----------------------------------")

data_name = "Reuters"
print(data_name)

print("knn")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["knn"])))
df_model = df_model[df_model.iloc[:,3] == "cm"]
df_model[4] = [e.diagonal().sum()/e.sum() for e in df_model[4]]
df_model = df_model.groupby([1,2])[4].mean().reset_index().sort_values(4,ascending=False)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("dt")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["dt"])))
df_model = df_model[df_model.iloc[:,2] == "cm"]
df_model[3] = [e.diagonal().sum()/e.sum() for e in df_model[3]]
df_model = df_model.groupby([1])[3].mean().reset_index().sort_values(3,ascending=False)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("rf")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["rf"])))
df_model = df_model[df_model.iloc[:,2] == "cm"]
df_model[3] = [e.diagonal().sum()/e.sum() for e in df_model[3]]
df_model = df_model.groupby([1])[3].mean().reset_index().sort_values(3,ascending=False)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])

print("gbm")
df_model = pd.DataFrame(dict_to_list(flatten_dict(cv_results[data_name]["gbm"])))
df_model = df_model[df_model.iloc[:,4] == "cm"]
df_model[5] = [e.diagonal().sum()/e.sum() for e in df_model[5]]
df_model = df_model.groupby([1,2,3])[5].mean().reset_index().sort_values(5,ascending=False)
cv_model_results_to_compare[data_name].append(df_model);print(df_model.iloc[:3,:])



# %% [markdown]
# ### Test Runs

# %% [markdown]
# - Best parameters for each model in each dataset are set in the final models. 

# %%
test_results = {e2:{e:{} for e in ["knn","dt","rf","gbm"]} for e2 in ['IDA2016', 'HillValley', 'Smartphone', 'YearPrediction', 'Reuters']}

data_name = "IDA2016"
model_name = "knn"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

model = KNeighborsClassifier(n_jobs=-1,n_neighbors=5,p=1)
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_valid)
pred = pred_proba.argmax(1)
test_results[data_name][model_name]["cm"] = confusion_matrix(y_valid, pred)
test_results[data_name][model_name]["ra"] = roc_auc_score(y_valid, pred_proba[:,1])
test_results[data_name][model_name]["prfs"] = precision_recall_fscore_support(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")



data_name = "HillValley"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

model = KNeighborsClassifier(n_jobs=-1,n_neighbors=4,p=1)
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_valid)
pred = pred_proba.argmax(1)
test_results[data_name][model_name]["cm"] = confusion_matrix(y_valid, pred)
test_results[data_name][model_name]["ra"] = roc_auc_score(y_valid, pred_proba[:,1])
test_results[data_name][model_name]["prfs"] = precision_recall_fscore_support(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")



data_name = "Smartphone"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

model = KNeighborsClassifier(n_jobs=-1,n_neighbors=5,p=2)
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_valid)
pred = pred_proba.argmax(1)
test_results[data_name][model_name]["cm"] = confusion_matrix(y_valid, pred)
test_results[data_name][model_name]["prfs"] = precision_recall_fscore_support(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")



data_name = "YearPrediction"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

model = KNeighborsRegressor(n_jobs=-1,n_neighbors=5,p=1)
model.fit(X_train, y_train)
pred = model.predict(X_valid)
test_results[data_name][model_name]["mse"] = mean_squared_error(y_valid, pred)
test_results[data_name][model_name]["mae"] = mean_absolute_error(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")



data_name = "Reuters"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

model = KNeighborsClassifier(n_jobs=-1,n_neighbors=1,p=1)
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_valid)
pred = pred_proba.argmax(1)
test_results[data_name][model_name]["cm"] = confusion_matrix(y_valid, pred)
test_results[data_name][model_name]["prfs"] = precision_recall_fscore_support(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")






data_name = "IDA2016"
model_name = "dt"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

m=5
model = DecisionTreeClassifier(random_state=0,ccp_alpha=0,min_samples_leaf=m,min_samples_split=m*2)
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_valid)
pred = pred_proba.argmax(1)
test_results[data_name][model_name]["cm"] = confusion_matrix(y_valid, pred)
test_results[data_name][model_name]["ra"] = roc_auc_score(y_valid, pred_proba[:,1])
test_results[data_name][model_name]["prfs"] = precision_recall_fscore_support(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")



data_name = "HillValley"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

m=1
model = DecisionTreeClassifier(random_state=0,ccp_alpha=0,min_samples_leaf=m,min_samples_split=m*2)
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_valid)
pred = pred_proba.argmax(1)
test_results[data_name][model_name]["cm"] = confusion_matrix(y_valid, pred)
test_results[data_name][model_name]["ra"] = roc_auc_score(y_valid, pred_proba[:,1])
test_results[data_name][model_name]["prfs"] = precision_recall_fscore_support(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")



data_name = "Smartphone"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

m=4
model =DecisionTreeClassifier(random_state=0,ccp_alpha=0,min_samples_leaf=m,min_samples_split=m*2)
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_valid)
pred = pred_proba.argmax(1)
test_results[data_name][model_name]["cm"] = confusion_matrix(y_valid, pred)
test_results[data_name][model_name]["prfs"] = precision_recall_fscore_support(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")



data_name = "YearPrediction"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

m=5
model = DecisionTreeRegressor(random_state=0,ccp_alpha=0,min_samples_leaf=m,min_samples_split=m*2)
model.fit(X_train, y_train)
pred = model.predict(X_valid)
test_results[data_name][model_name]["mse"] = mean_squared_error(y_valid, pred)
test_results[data_name][model_name]["mae"] = mean_absolute_error(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")



data_name = "Reuters"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

m=2
model = DecisionTreeClassifier(random_state=0,ccp_alpha=0,min_samples_leaf=m,min_samples_split=m*2)
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_valid)
pred = pred_proba.argmax(1)
test_results[data_name][model_name]["cm"] = confusion_matrix(y_valid, pred)
test_results[data_name][model_name]["prfs"] = precision_recall_fscore_support(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")






data_name = "IDA2016"
model_name = "rf"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

m="1"
model = RandomForestClassifier(random_state=0,n_jobs=3,max_features=max_features_map[m])
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_valid)
pred = pred_proba.argmax(1)
test_results[data_name][model_name]["cm"] = confusion_matrix(y_valid, pred)
test_results[data_name][model_name]["ra"] = roc_auc_score(y_valid, pred_proba[:,1])
test_results[data_name][model_name]["prfs"] = precision_recall_fscore_support(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")



data_name = "HillValley"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

m="sqrt/2"
model = RandomForestClassifier(random_state=0,n_jobs=3,max_features=max_features_map[m])
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_valid)
pred = pred_proba.argmax(1)
test_results[data_name][model_name]["cm"] = confusion_matrix(y_valid, pred)
test_results[data_name][model_name]["ra"] = roc_auc_score(y_valid, pred_proba[:,1])
test_results[data_name][model_name]["prfs"] = precision_recall_fscore_support(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")



data_name = "Smartphone"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

m="sqrt/2"
model = RandomForestClassifier(random_state=0,n_jobs=3,max_features=max_features_map[m])
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_valid)
pred = pred_proba.argmax(1)
test_results[data_name][model_name]["cm"] = confusion_matrix(y_valid, pred)
test_results[data_name][model_name]["prfs"] = precision_recall_fscore_support(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")



data_name = "YearPrediction"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

m="sqrt*2"
model = RandomForestRegressor(random_state=0,n_jobs=3,max_features=max_features_map[m])
model.fit(X_train, y_train)
pred = model.predict(X_valid)
test_results[data_name][model_name]["mse"] = mean_squared_error(y_valid, pred)
test_results[data_name][model_name]["mae"] = mean_absolute_error(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")



data_name = "Reuters"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

m="sqrt"
model = RandomForestClassifier(random_state=0,n_jobs=3,max_features=max_features_map[m])
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_valid)
pred = pred_proba.argmax(1)
test_results[data_name][model_name]["cm"] = confusion_matrix(y_valid, pred)
test_results[data_name][model_name]["prfs"] = precision_recall_fscore_support(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")






data_name = "IDA2016"
model_name = "gbm"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

md=-1
nt= 250
lr= 0.05
model = LGBMClassifier(random_state=0,n_jobs=-1,verbose=-1,max_depth=md,n_estimators=nt,learning_rate=lr)
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_valid.values)
pred = pred_proba.argmax(1)
test_results[data_name][model_name]["cm"] = confusion_matrix(y_valid, pred)
test_results[data_name][model_name]["ra"] = roc_auc_score(y_valid, pred_proba[:,1])
test_results[data_name][model_name]["prfs"] = precision_recall_fscore_support(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")



data_name = "HillValley"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

md= 1
nt= 100
lr= 0.05
model = LGBMClassifier(random_state=0,n_jobs=-1,verbose=-1,max_depth=md,n_estimators=nt,learning_rate=lr)
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_valid.values)
pred = pred_proba.argmax(1)
test_results[data_name][model_name]["cm"] = confusion_matrix(y_valid, pred)
test_results[data_name][model_name]["ra"] = roc_auc_score(y_valid, pred_proba[:,1])
test_results[data_name][model_name]["prfs"] = precision_recall_fscore_support(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")



data_name = "Smartphone"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

md= 3
nt= 500
lr= 0.2
model = LGBMClassifier(random_state=0,n_jobs=-1,verbose=-1,max_depth=md,n_estimators=nt,learning_rate=lr)
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_valid.values)
pred = pred_proba.argmax(1)
test_results[data_name][model_name]["cm"] = confusion_matrix(y_valid, pred)
test_results[data_name][model_name]["prfs"] = precision_recall_fscore_support(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")



data_name = "YearPrediction"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

md= 7
nt= 500
lr= 0.10
model = LGBMRegressor(random_state=0,n_jobs=-1,verbose=-1,max_depth=md,n_estimators=nt,learning_rate=lr)
model.fit(X_train, y_train)
pred = model.predict(X_valid.values)
test_results[data_name][model_name]["mse"] = mean_squared_error(y_valid, pred)
test_results[data_name][model_name]["mae"] = mean_absolute_error(y_valid, pred)

joblib.dump(test_results,"test_results1_a.pkl")



data_name = "Reuters"

train = data_for_models[data_name]["train"]
test = data_for_models[data_name]["test"]

y_train,X_train = train.iloc[:,0],train.iloc[:,1:]
y_valid,X_valid = test.iloc[:,0],test.iloc[:,1:]

md= -1
nt= 500
lr= 0.05
model = LGBMClassifier(random_state=0,n_jobs=-1,verbose=-1,max_depth=md,n_estimators=nt,learning_rate=lr)
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_valid.values)
pred = pred_proba.argmax(1)
test_results[data_name][model_name]["cm"] = confusion_matrix(y_valid, pred)
test_results[data_name][model_name]["prfs"] = precision_recall_fscore_support(y_valid, pred)


joblib.dump(test_results,"test_results1_a.pkl")




# %% [markdown]
# ### Test Results

# %% [markdown]
# For each model and dataset, test perforances are calculated below. The tables show the perforance metric averages of 5-folds for each parameter configuration of CV stage. Each table has an additional row showing the test performance.

# %%
model_names = ["knn","dt","rf","gbm"]
data_name = "IDA2016"
results_comparison_overall = pd.DataFrame()
for model_id in range(4):
    model_name = model_names[model_id]

    results_comparison = cv_model_results_to_compare[data_name][model_id].copy()
    results_comparison.insert(0,"type","valid")
    results_comparison = results_comparison.T.reset_index(drop=True).T
    test_row = np.append(np.append("test",results_comparison.iloc[0,1:-1].values),test_results[data_name][model_name]["ra"]).reshape(1,-1)
    results_comparison = pd.concat([results_comparison,pd.DataFrame(test_row)],axis=0)#.sort_values(len(results_comparison)-1,ascending=False)
    results_comparison = results_comparison.sort_values(results_comparison.shape[1]-1,ascending=False)

    results_for_overall = results_comparison.iloc[:,[0,-1]].T.reset_index(drop=True).T
    results_for_overall.insert(0,"model_name",model_name)
    results_for_overall.insert(0,"data_name",data_name)
    results_comparison_overall = pd.concat([results_comparison_overall,results_for_overall.copy()])

    print(data_name,model_name, "roc-auc comparison")
    print(results_comparison)
    print("------------------------------------\n------------------------------------")

data_name = "HillValley"
for model_id in range(4):
    model_name = model_names[model_id]

    results_comparison = cv_model_results_to_compare[data_name][model_id].copy()
    results_comparison.insert(0,"type","valid")
    results_comparison = results_comparison.T.reset_index(drop=True).T
    test_metric_value = test_results[data_name][model_name]["cm"]
    test_metric_value = test_metric_value.diagonal().sum()/test_metric_value.sum()
    test_row = np.append(np.append("test",results_comparison.iloc[0,1:-1].values),test_metric_value).reshape(1,-1)
    results_comparison = pd.concat([results_comparison,pd.DataFrame(test_row)],axis=0)#.sort_values(len(results_comparison)-1,ascending=False)
    results_comparison = results_comparison.sort_values(results_comparison.shape[1]-1,ascending=False)

    results_for_overall = results_comparison.iloc[:,[0,-1]].T.reset_index(drop=True).T
    results_for_overall.insert(0,"model_name",model_name)
    results_for_overall.insert(0,"data_name",data_name)
    results_comparison_overall = pd.concat([results_comparison_overall,results_for_overall.copy()])

    print(data_name,model_name, "accuracy comparison")
    print(results_comparison)
    print("------------------------------------\n------------------------------------")

data_name = "Smartphone"
for model_id in range(4):
    model_name = model_names[model_id]

    results_comparison = cv_model_results_to_compare[data_name][model_id].copy()
    results_comparison.insert(0,"type","valid")
    results_comparison = results_comparison.T.reset_index(drop=True).T
    test_metric_value = test_results[data_name][model_name]["cm"]
    test_metric_value = test_metric_value.diagonal().sum()/test_metric_value.sum()
    test_row = np.append(np.append("test",results_comparison.iloc[0,1:-1].values),test_metric_value).reshape(1,-1)
    results_comparison = pd.concat([results_comparison,pd.DataFrame(test_row)],axis=0)#.sort_values(len(results_comparison)-1,ascending=False)
    results_comparison = results_comparison.sort_values(results_comparison.shape[1]-1,ascending=False)

    results_for_overall = results_comparison.iloc[:,[0,-1]].T.reset_index(drop=True).T
    results_for_overall.insert(0,"model_name",model_name)
    results_for_overall.insert(0,"data_name",data_name)
    results_comparison_overall = pd.concat([results_comparison_overall,results_for_overall.copy()])

    print(data_name,model_name, "accuracy comparison")
    print(results_comparison)
    print("------------------------------------\n------------------------------------")


data_name = "YearPrediction"
for model_id in range(4):
    model_name = model_names[model_id]

    results_comparison = cv_model_results_to_compare[data_name][model_id].copy()
    results_comparison.insert(0,"type","valid")
    results_comparison = results_comparison.T.reset_index(drop=True).T
    test_row = np.append(np.append("test",results_comparison.iloc[0,1:-1].values),test_results[data_name][model_name]["mse"]).reshape(1,-1)
    results_comparison = pd.concat([results_comparison,pd.DataFrame(test_row)],axis=0)#.sort_values(len(results_comparison)-1,ascending=False)
    results_comparison = results_comparison.sort_values(results_comparison.shape[1]-1,ascending=True)

    results_for_overall = results_comparison.iloc[:,[0,-1]].T.reset_index(drop=True).T
    results_for_overall.insert(0,"model_name",model_name)
    results_for_overall.insert(0,"data_name",data_name)
    results_comparison_overall = pd.concat([results_comparison_overall,results_for_overall.copy()])

    print(data_name,model_name, "mse comparison")
    print(results_comparison)
    print("------------------------------------\n------------------------------------")

data_name = "Reuters"
for model_id in range(4):
    model_name = model_names[model_id]

    results_comparison = cv_model_results_to_compare[data_name][model_id].copy()
    results_comparison.insert(0,"type","valid")
    results_comparison = results_comparison.T.reset_index(drop=True).T
    test_metric_value = test_results[data_name][model_name]["cm"]
    test_metric_value = test_metric_value.diagonal().sum()/test_metric_value.sum()
    test_row = np.append(np.append("test",results_comparison.iloc[0,1:-1].values),test_metric_value).reshape(1,-1)
    results_comparison = pd.concat([results_comparison,pd.DataFrame(test_row)],axis=0)#.sort_values(len(results_comparison)-1,ascending=False)
    results_comparison = results_comparison.sort_values(results_comparison.shape[1]-1,ascending=False)

    results_for_overall = results_comparison.iloc[:,[0,-1]].T.reset_index(drop=True).T
    results_for_overall.insert(0,"model_name",model_name)
    results_for_overall.insert(0,"data_name",data_name)
    results_comparison_overall = pd.concat([results_comparison_overall,results_for_overall.copy()])

    print(data_name,model_name, "accuracy comparison")
    print(results_comparison)
    print("------------------------------------\n------------------------------------")

# %% [markdown]
# ## Summary

# %% [markdown]
# - In binary classification and regression problems, validation and test performances are similar.
# - In multi-class classification cases, the models perform poorly on the test set compared to the validation runs.
# 

# %%
valid_vs_test_overall = results_comparison_overall.copy()
valid_vs_test_overall[1] = valid_vs_test_overall[1]*((1-(valid_vs_test_overall["data_name"]=="YearPrediction"))*2-1)
valid_vs_test_overall = valid_vs_test_overall.groupby(["data_name","model_name",0]).max().reset_index()
valid_vs_test_overall = valid_vs_test_overall.pivot(index=["data_name",0],columns=["model_name"]).abs()
valid_vs_test_overall

# %% [markdown]
# Test and validation result for the same parameters are compared in the table above.
# - HillValley (accuracy)
#   - Although Lightgbm performs better in CV sets, test result of the same model is inferior compared to the other algorithms. The model parameters might lead to some overfitting despite the CV.
#   - RF is still robust for the test data. It has the best test result.
# - IDA2016 (roc-auc)
#   - All models performs similar in both stage.
#   - KNN is the worst among the algorithms. Applying some normalization would increase the performance of KNN.
# - Smartphone (accuracy)
#   - Test perfomances are very poor. A further investigation about the content of data is needed since all of the algorithms have the same levels of accuracy. Train and test sets' data regime might be different.
#   - LihtGBM and RF have the best performance in both validation and test sets.
# - YearPrediction (mse).
#   - Metrics are quite similar for both stages.
#   - KNN and DT seem to be insufficient for capturing the essence of this kind regression data.
# - Reuters (accuracy)
#   - Although accuracy levels seem low, regarding the number of classes (50) and used features (existence of 100 most used words in the article), the results are not cathastrophic.
#   - GBM and RF again have the best results. Similarly, RF is the best although it was inferior than the GBM model in the validation.



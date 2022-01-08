#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @作者：赵敬


# for load data
import csv

# for machine learning
import numpy as np
import pandas as pd
import mlflow
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection, preprocessing, feature_extraction, feature_selection, ensemble, linear_model, metrics, decomposition

# for plotting
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# .........read csv.........
# D:/文字资料-研/2021项目/Benchmark datasets/AB156/AB156/Features/AB156_Features_180.csv
# EMG+IMU
with open('D:/文字资料-研/2021项目/Benchmark datasets/AB156/AB156/Processed/AB156_Circuit_001_post.csv') as AB156_post:
    df1 = pd.read_csv(AB156_post).drop(['Right_Heel_Contact', 'Right_Heel_Contact_Trigger', 'Right_Toe_Off',
                                        'Right_Toe_Off_Trigger', 'Left_Heel_Contact', 'Left_Heel_Contact_Trigger',
                                        'Left_Toe_Off', 'Left_Toe_Off_Trigger'], axis=1)
# .........read some rows.........
    # for rows in reader:
    #     # row = rows
    #     print(rows[0])
# ................................
#     print(df1['Mode'].unique())     # 查看有几种类型
#     print(df1.groupby('Mode').size())     # 查看某列数据是否平衡
    # sns.countplot(df1['Mode'], label="Count")
    # plt.show()
    # df1.drop('Mode', axis=1).plot(kind='box', subplots=True, layout=(12, -1), sharex=False, sharey=False, figsize=(30, 30), title='Box Plot for each input variable')
    # # plt.savefig('fruits_box')
    # plt.show()
    # print(df1[['Mode', 'Left_Shank_Ay']].head())      # 得到的是dataFrame格式
# # ................................
    '''
        split datda
    '''
    df1_train, df1_test = model_selection.train_test_split(df1, test_size= 0.3)
    print("X_train shape:", df1_train.shape, "| X_test shape:", df1_test.shape)
    print("y_train mean:", round(np.mean(df1_train["Mode"]), 2), "| y_test mean:", round(np.mean(df1_test["Mode"]), 2))
    print(df1_train.shape[1], "features:", df1_train.drop("Mode",axis=1).columns.to_list())

    '''
        X 为合并后所有的特征列 
    '''
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # X1 = df1.iloc[0:, :30]
    # X2 = df1.iloc[0:, 45:52]
    # X = pd.concat([X1, X2], axis=1)
    X = scaler.fit_transform(df1_train)     # MinMaxScale 缩放特征值
    df1_sacale = pd.DataFrame(X, columns=df1_train.columns, index=df1_train.index)
    df1_sacale["Mode"] = df1_train["Mode"]
    print(df1_sacale.head())

    """
        Feature Selection ---correlation matrix
    """
    # corr_matrix = df1.copy()
    # for col in corr_matrix.columns:
    #     if corr_matrix[col].dtype == 'O':
    #         corr_matrix[col] = corr_matrix[col].factorize(sort=True)[0]         # ?
    #
    # corr_matrix = corr_matrix.corr(method="pearson")
    # sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=False, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)   # annot 表示是否显示数值
    # plt.title("pearson correlation")
    # # plt.show()

    """
    Automatic Feature Selection --- Lasso regularization/ ANOVA
    """
    # X = df1_train.drop("Mode", axis=1).values
    # y = df1_train["Mode"].values
    # feature_names = df1_train.drop("Mode", axis=1).columns
    #
    # # Anova
    # selector = feature_selection.SelectKBest(score_func=feature_selection.f_classif, k=10).fit(X, y)
    # anova_selected_features = feature_names[selector.get_support()]
    #
    # # Lasso regularization
    # selector = feature_selection.SelectFromModel(estimator=linear_model.LogisticRegression(C=1, penalty="l1", solver='liblinear'),
    #                                              max_features=10).fit(X, y)
    # lasso_selected_features = feature_names[selector.get_support()]
    #
    # # plot
    # df1_features = pd.DataFrame({"features": feature_names})
    # df1_features["anova"] = df1_features["features"].apply(lambda x: "anova" if x in anova_selected_features else 0)
    # df1_features["num1"] = df1_features["features"].apply(lambda x: 1 if x in anova_selected_features else 0)
    # df1_features["lasso"] = df1_features["features"].apply(lambda x: "lasso" if x in lasso_selected_features else "")
    # df1_features["num2"] = df1_features["features"].apply(lambda x: 1 if x in lasso_selected_features else 0)
    # df1_features["method"] = df1_features[["anova", "lasso"]].apply(lambda x: (str(x[0])+" " +str(x[1])).strip(), axis=1)
    # df1_features["selection"] = df1_features["num1"] + df1_features["num2"]
    # sns.barplot(y="features", x="selection", hue="method", data=df1_features.sort_values("selection", ascending=False), dodge=False)
    # plt.show()

    """
    Automatic Feature Selection --- Random forest: ensemble method consists of a number of decision trees
    features importance is computed from hou much each feature decreases the entropy in a tree 
    """
    # X = df1_train.drop("Mode", axis=1).values
    # y = df1_train["Mode"].values
    # feature_names = df1_train.drop("Mode", axis=1).columns.tolist()
    #
    # # importance
    # model = ensemble.RandomForestClassifier(n_estimators=100,
    #                                         criterion="entropy", random_state=0)
    # model.fit(X, y)
    # importances = model.feature_importances_
    #
    # # put in a pandas dft
    # dtf_importances = pd.DataFrame({"IMPORTANCE":importances,
    #                                "VARIABLE":feature_names}).sort_values("IMPORTANCE", ascending=False)
    # dtf_importances['cumsum'] = dtf_importances['IMPORTANCE'].cumsum(axis=0)
    # dtf_importances = dtf_importances.set_index("VARIABLE")
    #
    # # plot
    # fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
    # fig.suptitle("Features Importance", fontsize=20)
    # ax[0].title.set_text('variables')
    # dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(
    #     kind="barh", legend=False, ax=ax[0]).grid(axis="x")
    # ax[0].set(ylabel="")
    # ax[1].title.set_text('cumulative')
    # dtf_importances[["cumsum"]].plot(kind="line", linewidth=4, legend=False, ax=ax[1])
    # ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)),
    #           xticklabels=dtf_importances.index)
    # plt.xticks(rotation=70)
    # plt.grid(axis='both')
    # plt.show()

    """
        try to use less features ,note that before using test data for prediction 
        must preprocess it just like what did for train data
    """

    X_names = ["Waist_Az", "Left_Knee", "Right_Knee", "Left_Ankle", "Left_Thigh_Az", "Right_Thigh_Ax", "Left_Shank_Az"]
    X_train = df1_train[X_names].values
    y_train = df1_train["Mode"].values

    X_test = df1_test[X_names].values
    y_test = df1_test["Mode"].values

    """
        training
    """
    # call model
    model = ensemble.GradientBoostingClassifier()

    # define hyperparameters combinations to try
    param_dic = {'learning_rate': [0.15, 0.1, 0.05, 0.01, 0.005, 0.001],
                 # weighting factor for the corrections by new trees when added to the model
                 'n_estimators': [100, 250, 500, 750, 1000, 1250, 1500, 1750],  # number of trees added to the model
                 'max_depth': [2, 3, 4, 5, 6, 7],   # maximum depth of the tree
                 'min_samples_split': [2, 4, 6, 8, 10, 20, 40, 60, 100],    # sets the minimum number of samples to split
                 'min_samples_leaf': [1, 3, 5, 7, 9],   # sets the minimum number of samples to split
                 'max_features': [2, 3, 4, 5, 6, 7],    # square root of features is usually a good starting point
                 # the fraction of samples to be used for fitting the individual base learners...
                 # ...Values lower than 1 generally lead to a reduction of variance and an increase in bias.
                 'subsample': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
                 }

    # random search
    random_search = model_selection.RandomizedSearchCV(model,
                                                       param_distributions=param_dic, n_iter=1000,
                                                       scoring="accuracy").fit(X_train, y_train)
    print("Best Model parameters:", random_search.best_params_)
    print("Best Model mean accuracy:", random_search.best_score_)
    model = random_search.best_estimator_

#     X_std = StandardScaler
#     ().fit_transform(X)   # 标准化
#     print(X_std.shape)
#     y = df1['Mode']
#     # print(y)
#
#
# # ......... PCA 降维 ..............
#     n_components = 37    # 降维后维度
#     model = PCA(n_components=n_components)
#     model = model.fit(X_std)
#     X_dr = model.transform(X_std)       # 变化后的数据
#     # print(X_dr.shape)
#     explained_var = model.explained_variance_ratio_     # 获取贡献率
#     print(explained_var)
#     aggregate_explained_var = np.cumsum(model.explained_variance_ratio_)      # 累计贡献率
#     print(aggregate_explained_var)
# # ......... 3 维可视化数据              ？    ..............


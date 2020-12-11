import pandas as pd
import sklearn as sk
import numpy as np
from sklearn import model_selection
from sklearn import neighbors
from sklearn import ensemble
from sklearn import svm
import pathlib
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV


def main():
    cwd = pathlib.Path(__file__).parent.absolute()
    red_wine = pd.read_csv(str(cwd) + "/winequality-red.csv", delimiter=";")
    white_wine = pd.read_csv(str(cwd) + "/winequality-white.csv", delimiter=";")

    # print the predictors
    #print_individual_predictors(red_wine, "Red Wine")
    #print_individual_predictors(white_wine, "White Wine")

    # run the classifiers

    # HYPERPARAMETER TUNING -----
    #pick_best_knn(red_wine)
    #pick_best_ens(red_wine)

    # Finalized classifier calls
    print("Red wine dataset results:")
    knn_classifier(dataset=red_wine, dataset_name="red")
    ensemble_classifier(red_wine, "red")
    svm_classifier(red_wine, "red")

    print("\n\nWhite wine dataset results")
    knn_classifier(white_wine, "white")
    ensemble_classifier(white_wine, "white")
    svm_classifier(white_wine, "white")


# prints the weight of each of the predictors
def print_individual_predictors(dataset, dataset_name):
    predictors = dataset.columns
    quality = dataset["quality"]
    results = []

    for predictor in predictors:
        if (predictor != 'quality'):
            current_predictor = dataset[predictor]
            current_model = sm.OLS(quality, current_predictor).fit()
            results.append(current_model.fvalue)

            print("T-values: {}".format(current_model.tvalues))
            print("P-value: {}".format(current_model.f_pvalue))
            print("F-test: {} ".format(current_model.fvalue))
            print("F-test: {} ".format(current_model.rsquared))
            print("-----------------------\n")
        
    plt.style.use('ggplot')
    x = predictors
    x = x.drop(labels=['quality'])
    x_pos = [str(x[i]) for i in range(0, len(x))]

    plt.bar(x_pos, results, color='purple')
    plt.xlabel("Predictors")
    plt.ylabel("Quality")
    plt.title("Predictors vs Quality for "+dataset_name)

    plt.xticks(x_pos, x, rotation='vertical')

    plt.show()


# perform train-test split
def tt_split(dataset, rs):
    X = dataset.drop(columns=["quality"])
    y = dataset["quality"]

    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, random_state=rs)
    return X_train, X_test, y_train, y_test


# KNN ----------------------------------------------- #
def tt_split_knn(dataset, rs):
    df_model = dataset.copy()

    # Scale data to improve accuracy
    scaler = MinMaxScaler()
    features = ["fixed acidity", "density", "pH", "sulphates", "alcohol"]
    for feature in features:
        try:
            df_model[feature] = scaler.fit_transform(df_model[feature])
        except:
            df_model[feature] /= df_model[feature].max()

    X = df_model.drop(columns=["citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
                               "total sulfur dioxide", "volatile acidity", "quality"])
    y = df_model["quality"]

    pca = PCA()
    pca.fit(X)
    X = pca.transform(X)

    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, random_state=rs)
    return X_train, X_test, y_train, y_test


def pick_best_knn(dataset):
    n_neighbors = 50
    X_train, X_test, y_train, y_test = tt_split_knn(dataset, 42)

    knn = sk.neighbors.KNeighborsClassifier(n_neighbors)
    distributions = {
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [1, 2, 5, 10, 20, 30, 50],
    }
    clf = RandomizedSearchCV(knn, distributions, random_state=0)
    search = clf.fit(X_train, y_train)
    print(search.best_params_)
    print(search.best_estimator_)
    print(search.best_score_)


def knn_classifier(dataset, dataset_name):
    n_neighbors = 50
    # train-test split
    X_train, X_test, y_train, y_test = tt_split_knn(dataset, 42)
    
    # instantiate the model with 5 neighbors
    knn = sk.neighbors.KNeighborsClassifier(n_neighbors, weights='distance', leaf_size=2, algorithm='brute')
    # fit the model on the training data
    knn.fit(X_train, y_train)
    # see how the model performs
    print("KNN Score: {}".format(knn.score(X_test, y_test)))

    quality = knn.predict(X_test) # run the predictor on the test dataset and store results in array
    results = pd.DataFrame()
    results["quality"] = quality
    results.to_csv('./results/knn_'+dataset_name+'.csv', index=False) # write to csv

# ENSEMBLE ------------------------------------------- #
# evalute hyper-parameters
def pick_best_ens(dataset):
    X_train, X_test, y_train, y_test = tt_split(dataset, 50)

    ens = sk.ensemble.ExtraTreesClassifier()
    distributions = {
        'n_estimators': [10, 50, 100, 250, 500, 750, 1000],
        'criterion': ["gini", "entropy"],
        'max_depth': [1, 2, 5, 10, 25, 50, None],
        'max_features': ["auto", "sqrt", "log2", None],
        'bootstrap': [True, False],
        'class_weight': ["balanced", "balanced_subsample", None],
    }
    clf = RandomizedSearchCV(ens, distributions, random_state=0)
    search = clf.fit(X_train, y_train)
    print(search.best_params_)
    print(search.best_estimator_)
    print(search.best_score_)


# perform ensemble classification
def ensemble_classifier(dataset, dataset_name):
    # train-test split
    X_train, X_test, y_train, y_test = tt_split(dataset, 50)

    # instantiate the tree based cluster
    ens = sk.ensemble.ExtraTreesClassifier(n_estimators=100)
    ens.fit(X_train, y_train)
    print("Ensemble score: {}".format(ens.score(X_test, y_test)))

    new_ens = sk.ensemble.ExtraTreesClassifier(n_estimators=750, max_depth=25, criterion="entropy", bootstrap=True, class_weight="balanced_subsample")
    new_ens.fit(X_train, y_train)
    print("New Ensemble score: {}".format(new_ens.score(X_test, y_test)))

    quality = new_ens.predict(X_test) # run the predictor on the test dataset and store results in array
    results = pd.DataFrame()
    results["quality"] = quality
    results.to_csv('./results/ensemble_'+dataset_name+'.csv', index=False) # write to csv


# SVM ------------------------------------------------ #
def svm_classifier(dataset, dataset_name):
    # train-test split
    X_train, X_test, y_train, y_test = tt_split(dataset, 50)

    # instantiate svm classifier and print results
    svm = sk.svm.SVC(kernel='linear')
    svm.fit(X_train, y_train)
    print("SVM Score: {}".format(svm.score(X_test, y_test)))
    quality = svm.predict(X_test) # run the predictor on the test dataset and store results in array
    results = pd.DataFrame()
    results["quality"] = quality
    results.to_csv('./results/svm_'+dataset_name+'.csv', index=False) # write to csv

if __name__ == "__main__":
    main()

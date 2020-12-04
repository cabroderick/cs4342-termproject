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
from matplotlib.colors import ListedColormap

def main ():
    cwd = pathlib.Path(__file__).parent.absolute()
    red_wine = load_csv(str(cwd) + "/winequality-red.csv")
    white_wine = load_csv(str(cwd) + "/winequality-white.csv")

    # print the predictors
    #print_individual_predictors(red_wine)
    #print_individual_predictors(white_wine)

    # run the classifiers
    knn_classifier(red_wine, 50)
    ensemble_classifier(red_wine)
    svm_classifier(red_wine)

# prints the weight of each of the predictors
def print_individual_predictors(dataset):
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
    plt.title("Predictors vs Quality")

    plt.xticks(x_pos, x, rotation='vertical')

    plt.show()

# perform train-test split
def tt_split(dataset, rs):
    X = isolate_predictors(dataset)
    y = dataset.quality
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, random_state=rs)
    return X_train, X_test, y_train, y_test

# isolate only the predictors from the dataset
def isolate_predictors(dataset):
    return dataset.drop('quality',axis=1)

# perform knn classification
def knn_classifier(dataset, n_neighbors):
    # train-test split
    X_train, X_test, y_train, y_test = tt_split(dataset, 42)
    
    # instantiate the model with 5 neighbors
    knn = sk.neighbors.KNeighborsClassifier(n_neighbors)
    # fit the model on the training data
    knn.fit(X_train, y_train)
    # see how the model performs
    print(knn.score(X_test, y_test))

# perform ensemble classification
def ensemble_classifier(dataset):
    # train-test split
    X_train, X_test, y_train, y_test = tt_split(dataset, 50)

    # instantiate the tree based cluster
    ens = sk.ensemble.ExtraTreesClassifier(n_estimators=100)
    ens.fit(X_train, y_train)
    print(ens.score(X_test, y_test))

def svm_classifier(dataset):
    # train-test split
    X_train, X_test, y_train, y_test = tt_split(dataset, 50)

    # instantiate svm classifier and print results
    svm = sk.svm.SVC(kernel='linear')
    svm.fit(X_train, y_train)
    print(svm.score(X_test, y_test))

# loads a csv file
def load_csv(filename):
    return pd.read_csv(filename, sep=';')

if __name__ == "__main__":
    main()
import pandas as pd
import sklearn as sk
import numpy as np
from sklearn import model_selection
from sklearn import neighbors
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
    #knn_classifier(dataset=red_wine, n_neighbors=50)
    ensemble_classifier(dataset=red_wine)

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

def isolate_predictors(dataset, color):
    if color == "red":
        return dataset.drop(['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','sulphates','quality'], axis=1)
        # isolate predictors for red wine dataset
    if color == "white":
        pass
        # isolate predictors for white wine dataset

def knn_classifier(dataset, n_neighbors):
    # split data into modeling and target variables
    X = isolate_predictors(dataset, "red")
    y = dataset.quality

    # train-test split
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, random_state=42)
    # instantiate the model with 5 neighbors
    knn = sk.neighbors.KNeighborsClassifier(n_neighbors)
    # fit the model on the training data
    knn.fit(X_train, y_train)
    # see how the model performs
    print(knn.score(X_test, y_test))

    # set up the color map
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])

    # # calculate min, max and limits
    # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    # np.arange(y_min, y_max, h))

    # # predict class using data and kNN classifier
    # Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # plt.figure()
    # plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # # Plot also the training points
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    # plt.title("KNN Classification (k = %i)" % (n_neighbors))
    # plt.show()
    
def ensemble_classifier(dataset):
    # split data into modeling and target variables
    X = dataset.drop(['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','sulphates','quality'], axis=1)
    y = dataset.quality

    # train-test split
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, random_state=42)

    # instantiate the tree based cluster
    ens = sk.ensemble.ExtraTreesClassifier(n_estimators=100)
    ens.fit(X_train, y_train)
    print(ens.score(X_test, y_test))

def load_csv(filename):
    return pd.read_csv(filename, sep=';')

if __name__ == "__main__":
    main()
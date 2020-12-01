import pandas as pd
import sklearn as sk
from sklearn import model_selection
from sklearn import neighbors
import pathlib
import statsmodels.api as sm
import matplotlib.pyplot as plt

def main ():
    cwd = pathlib.Path(__file__).parent.absolute()
    red_wine = load_csv(str(cwd) + "/winequality-red.csv")
    white_wine = load_csv(str(cwd) + "/winequality-white.csv")

    #print_individual_predictors(red_wine)
    #print_individual_predictors(white_wine)
    knn_classifier(red_wine)

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

def knn_classifier(dataset):
    # split data into modeling and target variables
    X = dataset.drop(['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','sulphates','quality'], axis=1)
    y = dataset.quality

    # train-test split
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, random_state=42)
    # instantiate the model with 5 neighbors
    knn = sk.neighbors.KNeighborsClassifier(n_neighbors=5)
    # fit the model on the training data
    knn.fit(X_train, y_train)
    # see how the model performs
    print(knn.score(X_test, y_test))

def load_csv(filename):
    return pd.read_csv(filename, sep=';')

if __name__ == "__main__":
    main()
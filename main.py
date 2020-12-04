import pandas as pd
import sklearn as sk
import numpy as np
from sklearn import model_selection
from sklearn import neighbors
from sklearn import ensemble
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
    #ensemble_classifier(dataset=red_wine)

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
    visualize(X, y, knn, "KNN")
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

# model = model
# plot_title = string name (for descriptive plots)
def visualize(X, y, model, plot_title):
    h = 0.02

    # set up the color map (3-most red, 9- most green
    # cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_light = ListedColormap(['#FF0000', '#D42B00', '#AA5500', '#808000', '#55AA00', '#2BD400', '#00FF00'])
    cmap_bold = ListedColormap(['#FF0000', '#D42B00', '#AA5500', '#808000', '#55AA00', '#2BD400', '#00FF00'])

    # calculate min, max and limits

    # whats broken is that the X range X[:, 0] does not work for some reason. This logic will also only
    # work with two parameters as we are graphing a 2d grid at the moment. make it work with just two predictors and then
    # we can go from there. its creating the bounds of the plot by looking at the range of values for the two predictors
    # it then overlays the predictions onto this range
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # predict class using data and kNN classifier
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("{}".format(plot_title))
    plt.show()

# loads a csv file
def load_csv(filename):
    return pd.read_csv(filename, sep=';')

if __name__ == "__main__":
    main()
import pandas as pd
import sklearn as sk
import pathlib
import statsmodels.api as sm
import matplotlib.pyplot as plt

def main ():
    cwd = pathlib.Path(__file__).parent.absolute()
    red_wine = load_csv(str(cwd) + "/winequality-red.csv")
    white_wine = load_csv(str(cwd) + "/winequality-white.csv")

    print_individual_predictors(red_wine)

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


def load_csv(filename):
    return pd.read_csv(filename, sep=';')

if __name__ == "__main__":
    main()
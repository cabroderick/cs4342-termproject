import pandas as pd
import sklearn as sk

def main ():
    redWine = load_csv("winequality-red.csv")
    whiteWine = load_csv("winequality-white.csv")

def load_csv(filename):
    return pd.read_csv(filename)

if __name__ == "__main__":
    main()
import pandas as pd
import joblib
from helper import data_preperation_for_model
from config import hyperparameter_optimization


def main():
    dataframe = pd.read_excel("workers_dataset.xlsx", engine='openpyxl')
    X, y, dataframe_final = data_preperation_for_model(dataframe)
    final_model = hyperparameter_optimization(X, y, cv=5)
    joblib.dump(final_model, "final_model.pkl")
    dataframe_final.to_csv("final_dataframe.csv")
    return final_model


if __name__ == "__main__":
    main()

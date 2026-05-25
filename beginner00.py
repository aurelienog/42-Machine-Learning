#!/usr/bin/env python3

import pandas
import seaborn  # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
import matplotlib.pyplot as plt


def plot(df):
    """
    Display dataset and regression line.
    """

    seaborn.lmplot(
        x="YearsExperience",
        y="Salary",
        data=df,
        height=7,
        aspect=1.5
    )

    plt.show()

def train_model(df):
    """
    Train a simple linear regression model.
    """

    X = df[["YearsExperience"]]
    y = df["Salary"]

    regression = LinearRegression()

    regression.fit(X, y)

    return regression

def main():

    salary = pandas.read_csv("Salary_dataset.csv")

    seaborn.set_theme()

    regression = train_model(salary)

    # ex01
    plot(salary)

    # ex02
    new_data = pandas.DataFrame({"YearsExperience": [10]})

    prediction = regression.predict(new_data)

    print(
        "Predicted salary for 10 years of experience :",
        prediction[0]
    )

if __name__ == "__main__":
    main()

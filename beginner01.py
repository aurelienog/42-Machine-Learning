#!/usr/bin/env python3

import matplotlib.pyplot as plt  # type: ignore
import numpy as np    # type: ignore
import duckdb  # type: ignore


def fit_linear_regression(x, y):
    """
    Compute the coefficients of a simple linear regression using
    the closed-form least squares solution.

    Args:
        x: Independent variable values.
        y: Dependent variable values.

    Returns:
        tuple: (b0, b1) where:
            - b0 is the intercept.
            - b1 is the slope.
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    b1 = np.sum((x-x_mean)*(y-y_mean)) / np.sum((x-x_mean)**2)

    b0 = y_mean - b1*x_mean

    return b0,b1


def predict(x, b0, b1):
    """
    Predict target values using a linear regression model.

    Args:
        x: Input feature values.
        b0: Regression intercept.
        b1: Regression slope.

    Returns:
        Predicted value(s).
    """
    return b0 + b1 * x


def plot_training_iteration(x, y, b0, b1, i):
    """
    Predict target values using a linear regression model.

    Args:
        x: Input feature values.
        b0: Regression intercept.
        b1: Regression slope.

    Returns:
        Predicted value(s).
    """
    plt.figure()

    plt.scatter(x,y,color="blue")

    x_line = np.linspace(min(x),max(x),100)
    y_line = predict(x_line,b0,b1)

    plt.plot(x_line,y_line,color="purple")

    plt.title(f"Iteration {i}")

    plt.show()


def train_gradient_descent(x, y):
    """
    Train a simple linear regression model using gradient descent.

    Displays the evolution of the regression line at selected
    iterations.

    Args:
        x: Independent variable values.
        y: Dependent variable values.

    Returns:
        tuple: (b0, b1) trained regression coefficients.
    """
    b0 = 0
    b1 = 0

    learning_rate = 0.01
    num_iters = 1001
    n = len(x)

    for i in range(1, num_iters):
        if i in [0,1,2,3,4,1000]:
            plot_training_iteration(x, y, b0, b1, i)

        y_pred = predict(x,b0,b1)

        error = y_pred - y

        db0 = (1/n)*np.sum(error)
        db1 = (1/n)*np.sum(error*x)

        b0 -= learning_rate*db0
        b1 -= learning_rate*db1

    return b0,b1


def plot_dataset(title, x, y) -> None:
    """
    Display a scatter plot of the dataset.

    Args:
        title: Plot title.
        x: Tuple containing x-axis label and values.
        y: Tuple containing y-axis label and values.
    """
    x_label, x_data = x
    y_label, y_data = y
    plt.scatter(x_data, y_data)

    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)

    plt.show()


def main() -> None:
    """
    Load the salary dataset, display the data, train a linear
    regression model using gradient descent, and predict salaries
    for 10 and 15 years of experience.
    """

    salary_df = duckdb.read_csv("Salary_dataset.csv").df()

    x = ("YearsExperience", salary_df["YearsExperience"])
    y = ("Salary", salary_df["Salary"])
    title = "Years of Experience vs Salary"

    # print("ex00")
    plot_dataset(title, x, y)

    # print("ex01")
    b0,b1 = train_gradient_descent(x[1],y[1])

    print("ex02")
    salary_10 = predict(10,b0,b1)
    salary_15 = predict(15,b0,b1)

    print(f"Predicted salary for 10 years of experience {salary_10}")
    print(f"Predicted salary for 15 years of experience {salary_15}")


if __name__ == "__main__":
    main()

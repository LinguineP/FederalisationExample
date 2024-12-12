import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def print_data_info(data):
    print(data.head())
    print(f"Shape of data: {data.shape}")
    print(data.info())

def clean_data(data):
    columns_to_drop = ["housing_median_age", "households", "total_bedrooms", "longitude", 
                        "latitude", "total_rooms", "population"]
    return data.drop(columns=columns_to_drop)

def evaluate_model(y_test, y_predicted):
    mse = mean_squared_error(y_test, y_predicted)
    rmse = np.sqrt(mse) 
    mae = mean_absolute_error(y_test, y_predicted)
    r2 = r2_score(y_test, y_predicted)
    
    
    print("####################################")
    print(f"MSE: {mse:.4f} \nRMSE: {rmse:.4f} \nMAE: {mae:.4f} \nR²: {r2:.4f}")
    print("####################################")
    
    return rmse, r2

def linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    evaluate_model(y_test, y_predicted)
    
    return y_predicted

def plot_linear_regression(X_train, y_train, X_test, y_predicted):
    plt.scatter(X_train['median_income'], y_train, s=10)
    plt.xlabel('median_income')
    plt.ylabel('median_house_value')
    plt.plot(X_test['median_income'], y_predicted, color='r')
    plt.show()

def plot_residuals(y_test, y_predicted):
    residual = y_test - y_predicted
    sns.residplot(x=y_predicted, y=residual, lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plt.show()

def main():
    data = pd.read_csv("BaseCode/input/housing.csv")
    print_data_info(data)
    data = clean_data(data)
    X = data.drop("median_house_value", axis=1)
    y = data["median_house_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    y_predicted_lr = linear_regression(X_train, y_train, X_test, y_test)
    plot_linear_regression(X_train, y_train, X_test, y_predicted_lr)
    plot_residuals(y_test, y_predicted_lr)
    print("Model R2 score:", r2_score(y_test, y_predicted_lr))

if __name__ == "__main__":
    main()

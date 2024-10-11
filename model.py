import csv
import numpy as np
import joblib
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def plot_result(y_test, y_pred, mse, r2, sample_size=100):
    # Increase plot size
    plt.figure(figsize=(14, 8))

    # Reduce the amount of data visualized
    if len(y_test) > sample_size:
        indices = np.random.choice(len(y_test), sample_size, replace=False)
        y_test = y_test[indices]
        y_pred = y_pred[indices]

    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Comparison between True Values and Predictions')
    plt.text(0.05, 0.95, f'MSE: {mse:.2f}\nR²: {r2:.2f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

    # Line Plot
    plt.figure(figsize=(14, 8))
    plt.plot(y_test, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='red', linestyle='dashed')
    plt.xlabel('Sample Index')
    plt.ylabel('Prime Number')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.text(0.05, 0.95, f'MSE: {mse:.2f}\nR²: {r2:.2f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

    # Error histogram
    errors = y_test - y_pred
    plt.figure(figsize=(14, 8))
    plt.hist(errors, bins=50, alpha=0.75)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Prediction Errors')
    plt.text(0.05, 0.95, f'MSE: {mse:.2f}\nR²: {r2:.2f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.show()


def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def train_model():
    data = []
    with open('data.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append([int(row[0]), int(row[1]), int(row[2])])

    data = np.array(data)
    random.shuffle(data)

    x = data[:, [0, 2]]  # rank and interval
    y = data[:, 1]  # num

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(x_train, y_train)
    joblib.dump(model, 'random_forest_model.pkl')

    y_pred = model.predict(x_test)

    mse, r2 = evaluate_model(y_test, y_pred)

    plot_result(y_test, y_pred, mse, r2)

def import_model():
    return joblib.load('random_forest_model.pkl')

def predict(model, rank, interval):
    return model.predict([[rank, interval]])[0]

if __name__ == '__main__':
    train_model()
    model = import_model()
    print(predict(model, 100, 10))

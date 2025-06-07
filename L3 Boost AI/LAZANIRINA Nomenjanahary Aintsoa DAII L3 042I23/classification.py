import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

def load_subset_mnist(start=100, end=150, train_limit=110):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X_all = mnist.data.astype('float32') / 255.0
    y_all = mnist.target.astype(int)

    X_subset = X_all[start:end]
    y_subset = y_all[start:end]

    split = train_limit - start
    X_train, y_train = X_subset[:split], y_subset[:split]
    X_test, y_test = X_subset[split:], y_subset[split:]

    return X_train, y_train, X_test, y_test

def train_perceptron(X_train, y_train):
    model = Perceptron(max_iter=1000, eta0=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f" Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

def show_some_predictions(model, X_test, y_test):
    for i in range(len(X_test)):
        img = X_test[i].reshape(28, 28)
        prediction = model.predict([X_test[i]])[0]

        plt.imshow(img, cmap='gray')
        plt.title(f"Vrai: {y_test[i]} | Prédit: {prediction}")
        plt.axis('off')
        plt.show()

def main():
    X_train, y_train, X_test, y_test = load_subset_mnist(100, 150, 110)
    model = train_perceptron(X_train, y_train)
    evaluate(model, X_test, y_test)
    show_some_predictions(model, X_test, y_test)

if __name__ == "__main__":
    main()

import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import random

MODEL_PATH = "model_mnist.h5"

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

def build_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size
    )
    return history

def evaluate_model(model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"✅ Accuracy sur les données de test : {acc*100:.2f}%")

def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='Train acc')
    plt.plot(history.history['val_accuracy'], label='Test acc')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

def save_model(model, path=MODEL_PATH):
    model.save(path)
    print(f"💾 Modèle sauvegardé à l'emplacement : {path}")

def load_saved_model(path=MODEL_PATH):
    model = load_model(path)
    print(f"📥 Modèle chargé depuis : {path}")
    return model

def test_random_image(model, x_test, y_test):
    index = random.randint(0, len(x_test) - 1)
    image = x_test[index]
    label = np.argmax(y_test[index])

    prediction = model.predict(image.reshape(1, 28, 28, 1))
    predicted_label = np.argmax(prediction)

    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"Vrai : {label}, Prédit : {predicted_label}")
    plt.axis('off')
    plt.show()

def main():
    x_train, y_train, x_test, y_test = load_data()

    if os.path.exists(MODEL_PATH):
        model = load_saved_model()
    else:
        print("🔧 Aucun modèle trouvé, entraînement en cours...")
        model = build_model()
        history = train_model(model, x_train, y_train, x_test, y_test)
        evaluate_model(model, x_test, y_test)
        plot_training_history(history)
        save_model(model)

    # Test sur une image aléatoire
    test_random_image(model, x_test, y_test)

if __name__ == "__main__":
    main()

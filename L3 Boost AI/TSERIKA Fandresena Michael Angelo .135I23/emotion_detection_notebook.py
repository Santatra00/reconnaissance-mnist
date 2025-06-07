# %matplotlib inline
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import unicodedata
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from tabulate import tabulate

def clean_text(text):
    text = unicodedata.normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s]', '', text)
    return text

phrases = [
    "Je suis absolument ravi de cette nouvelle !",
    "Quelle journée magnifique, je suis aux anges !",
    "Je me sens tellement heureux avec mes amis.",
    "C’est incroyable, je ne pourrais pas être plus content !",
    "J’ai le cœur léger, tout va bien aujourd’hui.",
    "Je me sens abattu, rien ne va comme je veux.",
    "Aujourd’hui, je suis vraiment mélancolique.",
    "Je me sens seul, personne ne semble comprendre.",
    "Cette nouvelle m’a brisé le cœur.",
    "Je suis triste, tout me semble gris.",
    "Je suis furieux, c’est inacceptable !",
    "Ça me met hors de moi, quelle injustice !",
    "Je ne supporte plus cette situation, je suis en rage !",
    "Tout m’énerve aujourd’hui, ras-le-bol !",
    "Je suis tellement agacé par ce comportement !"
]
emotions = ["joie", "joie", "joie", "joie", "joie", "tristesse", "tristesse", "tristesse", "tristesse", "tristesse", "colère", "colère", "colère", "colère", "colère"]

phrases = [clean_text(phrase) for phrase in phrases]

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(emotions)
labels = np.array(label_tokenizer.texts_to_sequences(emotions)).flatten() - 1
classes = list(label_tokenizer.word_index.keys())

vocab_size = 100
max_length = 10
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(phrases)
sequences = tokenizer.texts_to_sequences(phrases)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)

model.save('emotion_model.h5')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Précision entraînement')
plt.plot(history.history['val_accuracy'], label='Précision validation')
plt.title('Précision du modèle')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perte entraînement')
plt.plot(history.history['val_loss'], label='Perte validation')
plt.title('Perte du modèle')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()
plt.show()

def predict_emotion(text):
    cleaned_text = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    pred = model.predict(padded, verbose=0)
    return classes[np.argmax(pred)]

def analyze_file(file_path):
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier '{file_path}' n'existe pas.")
        return
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]
        if not lines:
            print("Erreur : Le fichier est vide.")
            return
        results = [(phrase, predict_emotion(phrase)) for phrase in lines]
        print(tabulate(results, headers=["Phrase", "Émotion prédite"], tablefmt="grid"))
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")

test_phrase = "je suis super content"
print(f"Phrase : {test_phrase}")
print(f"Émotion prédite : {predict_emotion(test_phrase)}")

file_path = "phrases.txt"
analyze_file(file_path)
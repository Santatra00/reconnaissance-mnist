import numpy as np
# Fonction d'activation : step (seuil)
activation = lambda z: 1 if z >= 0 else 0

# Fonction de prédiction : dot + bias -> activation
predict = lambda x, w, b: activation(np.dot(x, w) + b)

# Données d'apprentissage pour AND
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 0, 0, 1])

# Poids et biais appris pour AND (à la main ou via entraînement)
w = np.array([1, 1])
b = -1.2

# Prédictions pour AND
results = list(map(lambda x: predict(x, w, b), X))
print("AND :", results) 
import random

# Données d'entraînement pour le AND logique
X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
Y = [0, 0, 0, 1]

# Initialisation des poids et biais
w = [random.uniform(-1, 1), random.uniform(-1, 1)]
b = random.uniform(-1, 1)

# Fonction d'activation : fonction seuil
def step(x):
    return 1 if x >= 0 else 0

# Apprentissage
learning_rate = 0.1
epochs = 20

for epoch in range(epochs):
    total_error = 0

    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        # Prédiction
        z = x[0] * w[0] + x[1] * w[1] + b
        y_pred = step(z)

        # Erreur
        error = y - y_pred
        total_error += abs(error)

        # Mise à jour des poids et biais
        w[0] += learning_rate * error * x[0]
        w[1] += learning_rate * error * x[1]
        b += learning_rate * error

    print(f"Epoch {epoch+1} - Erreur totale : {total_error}")

print("\nPoids appris :", w)
print("Biais appris :", b)

# Test final
print("\nTest du perceptron :")
for x in X:
    z = x[0] * w[0] + x[1] * w[1] + b
    y_pred = step(z)
    print(f"Entrée : {x} → Prédit : {y_pred}")

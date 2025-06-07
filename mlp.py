import numpy as np

# Fonction d'activation ReLU (pour la couche cachée)
def relu(z):
    return np.maximum(0, z)

# Fonction d'activation sigmoid (pour la sortie)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Données d'entrée (exemple 2 features)
X = np.array([0.6, 0.9])

# Poids et biais couche cachée (2 neurones dans la couche cachée)
W1 = np.array([[0.2, 0.8], 
               [0.5, -0.91]])
b1 = np.array([0.1, -0.3])

# Poids et biais couche de sortie (1 neurone de sortie)
W2 = np.array([0.7, -1.2])
b2 = 0.3

# Propagation avant
def forward_pass(X):
    # Couche cachée
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    
    # Couche de sortie
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    return A2

# Calcul sortie
output = forward_pass(X)
print(f"Sortie du réseau : {output:.4f}")
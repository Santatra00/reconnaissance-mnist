import numpy as np
import tensorflow as tf

# Vérification de la primalité
def est_premier(nombre):
    nombre = int(nombre) if np.isscalar(nombre) else int(nombre[0])
    if nombre < 2:
        return 0
    for i in range(2, int(np.sqrt(float(nombre))) + 1):  
        if nombre % i == 0:
            return 0
    return 1

# Données d'entraînement
nombres = np.arange(2, 1000).reshape(-1, 1).astype(np.float32)
etiquettes_classification = np.array([est_premier(n) for n in nombres], dtype=np.float32).reshape(-1, 1)

# Extraction des caractéristiques
caracteristiques = np.hstack([
    nombres, nombres % 2, nombres % 3, nombres % 5, nombres % 7
]).astype(np.float32)

# Modèle Perceptron
parametres = tf.Variable(tf.random.normal([caracteristiques.shape[1], 1]))
biais = tf.Variable(tf.random.normal([1]))

def perceptron_simple(entree, parametres, biais):
    return tf.nn.sigmoid(tf.matmul(entree, parametres) + biais)

optimiseur = tf.optimizers.Adam()

# Entraînement classification
for i in range(500):
    with tf.GradientTape() as calcul_gradients:
        sortie = perceptron_simple(caracteristiques, parametres, biais)
        perte = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=sortie, labels=etiquettes_classification))
    
    gradients = calcul_gradients.gradient(perte, [parametres, biais])
    optimiseur.apply_gradients(zip(gradients, [parametres, biais]))

# fonction de classification
def classifier_premier(liste_nombres):
    liste_nombres = np.array(liste_nombres, dtype=np.float32).reshape(-1, 1)
    nouvelles_caracteristiques = np.hstack([
        liste_nombres, liste_nombres % 2, liste_nombres % 3, liste_nombres % 5, liste_nombres % 7
    ]).astype(np.float32)

    predictions = perceptron_simple(nouvelles_caracteristiques, parametres, biais).numpy()
    return [("Premier" if p >= 0.5 else "Non Premier") for p in predictions]

#fonction de prédiction (probabilité d'être premier)
def predire_probabilite_premier(liste_nombres):
    liste_nombres = np.array(liste_nombres, dtype=np.float32).reshape(-1, 1)
    nouvelles_caracteristiques = np.hstack([
        liste_nombres, liste_nombres % 2, liste_nombres % 3, liste_nombres % 5, liste_nombres % 7
    ]).astype(np.float32)

    predictions = perceptron_simple(nouvelles_caracteristiques, parametres, biais).numpy()
    return [f"Probabilité d'être premier: {float(p):.2f}" for p in predictions.flatten()]


# echantillon de donnees

# Test Classification
test_classification = [1, 50, 101]
resultats_classification = classifier_premier(test_classification)
print("Classification :", resultats_classification)

test_prediction = [1, 50, 101]
resultats_prediction = predire_probabilite_premier(test_prediction)
print("Prédiction :", resultats_prediction)
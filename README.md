
# 🧠 Projet IA – Reconnaissance de chiffres manuscrits avec CNN (MNIST)

Ce projet implémente un **réseau de neurones convolutif (CNN)** en utilisant **TensorFlow / Keras** pour reconnaître les chiffres manuscrits de la base de données **MNIST**.

Le modèle est capable d’atteindre une **précision supérieure à 98%**, et inclut des fonctionnalités de sauvegarde/chargement du modèle ainsi que la prédiction d'une image aléatoire du jeu de test.

---

## 📁 Structure du projet

```bash
.
├── model_mnist.h5       # (Créé automatiquement après l'entraînement)
├── main.py              # Script principal contenant le modèle et la logique
└── README.md            # Ce fichier
```

---

## 🔧 Fonctionnalités

- Téléchargement et prétraitement des données MNIST
    
- Construction d’un CNN avec deux couches convolutives
    
- Entraînement du modèle avec affichage de la précision
    
- Évaluation sur le jeu de test
    
- Sauvegarde automatique du modèle (`.h5`)
    
- Chargement du modèle déjà entraîné
    
- Prédiction d’une image aléatoire avec visualisation
    
- Graphique d’évolution de la précision sur les époques
    

---

## 🧠 Architecture du modèle

- `Conv2D` (32 filtres) + `ReLU` + `MaxPooling2D`
    
- `Conv2D` (64 filtres) + `ReLU` + `MaxPooling2D`
    
- `Flatten`
    
- `Dense` (128 neurones) + `ReLU`
    
- `Dense` (10 neurones) + `Softmax`
    

---

## ▶️ Lancer le projet

### Prérequis

Assurez-vous d’avoir Python 3.x et les bibliothèques suivantes :

```bash
pip install tensorflow matplotlib numpy
```

### Exécution

```bash
python main.py
```

- Si aucun modèle n’est détecté (`model_mnist.h5`), un entraînement sera lancé.
    
- Sinon, le modèle existant sera chargé automatiquement.
    

---

## 📊 Exemple de sortie

- Affichage de la précision sur les données de test :
    
    ```
    ✅ Accuracy sur les données de test : 98.75%
    ```
    
- Courbe d’évolution de l’accuracy sur les époques
    
- Affichage d’une image test aléatoire :
    
    ```
    Vrai : 7, Prédit : 7
    ```
    

---

## 💡 Idées d’améliorations

Voici quelques pistes pour rendre ce projet encore plus impressionnant :

- Ajouter **Dropout** pour éviter le surapprentissage
    
- Permettre l’**entraînement sur GPU** si disponible
    
- Tester sur le dataset **Fashion-MNIST**
    
- Sauvegarder et afficher la **matrice de confusion**
    
- Utiliser **TensorBoard** pour la visualisation
    

---

## 🏆 Objectif

Ce projet montre ta capacité à :

- Comprendre et implémenter un CNN de A à Z
    
- Gérer l'entraînement, l'évaluation et la persistance d’un modèle
    
- Visualiser les résultats et expliquer ton code proprement
    

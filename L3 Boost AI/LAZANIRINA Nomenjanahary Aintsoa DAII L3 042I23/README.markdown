# Classification de chiffres manuscrits avec un perceptron

Ce projet utilise un **perceptron** pour classer des chiffres manuscrits (0 à 9) à partir du dataset **MNIST**. Il charge un sous-ensemble de 50 images, entraîne le modèle sur 10 images, teste sur 40 images, et affiche la précision, un rapport de classification, et 5 exemples de prédictions.

## Dépendances

Pour exécuter ce code, vous devez installer les bibliothèques Python suivantes :

- **Python** : Version 3.6 ou supérieure
- **NumPy** : Pour manipuler les tableaux de données (images).
  ```bash
  pip install numpy
  ```
- **scikit-learn** : Pour le modèle perceptron et les métriques d’évaluation.
  ```bash
  pip install scikit-learn
  ```
- **Matplotlib** : Pour afficher les images et les prédictions.
  ```bash
  pip install matplotlib
  ```

## Comment exécuter le code

1. Installez les dépendances listées ci-dessus.
2. Placez le fichier Python (par exemple, `classification.py`) dans un dossier.
3. Exécutez le script avec :
   ```bash
   python classification.py
   ```
4. Le programme affichera :
   - La précision du modèle (en pourcentage).
   - Un rapport détaillé des performances pour chaque chiffre (0 à 9).
   - 5 images de test avec leurs vraies étiquettes et les prédictions.


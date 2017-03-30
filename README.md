# Boson

## Note :

Il est necessaire d'enlever le header des fichiers train et test.
 
## GridSearch.py

Les parametres à gridsearcher sont dans la categorie 'Parametres'. 

## Best_model.py

Second meilleur modele (mais beaucoup plus rapide que le meilleur modele pour des performances a peine moindre.
Apres lancement, crée un fichier Submission.csv au format accepté par kaggle

## First_layer_restricted.py

Meilleur modele, mais 8 fois plus lent que le modele precedent.


## BDT.py

Fichier pour les arbres de décision boostés.

- Import des données et séparation en train/test
- Définition de la fonction AMS

Méthodes Gradient Boosting, Random Forest et Adaboost. Pour chaque méthode :
- implémentation naïve
- grid search sur les paramètres n_estimators et éventuellement profondeur de l'arbre
- Utilisation du meilleur classifieur sur les données de test de la compétition
- Création d'un fichier à soumettre sur Kaggle.com

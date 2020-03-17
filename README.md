# kaggle



## Introduction

Solution to kaggle "House Prices: Advanced Regression Techniques" competition.

## Utilisation du code

### Création d'un environnement virtuel

Pour utiliser le code de ce dépôt, il faut d'abord créer un environnement virtuel avec *virtualenv* :


+ Se placer à la racine du projet
+ Installer virtualenv via pip : `pip install virtualenv`
+ Créer l'environnement virtuel : `virtualenv .env`
+ activer l'envirennoment virtuel : `source .env/bin/activate`
+ installer les librairies : `pip install -r requirements.txt`


### Lancement de l'entraînement du modèle

Pour faire tourner le script principal *main.py* :

+ `python -m blueprint` : permet d'entrainer le modèle et d'obtenir une évaluation des performances sur le jeu d'entrainement


### Lancement des tests

Pour lancer l'intégralité des tests unitaire, il suffit de se placer à la racine du dossier et de lancer la commande : 

+ `python -m unittest` 

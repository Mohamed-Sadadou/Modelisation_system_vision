# Modelisation_system_vision
This is a notebook for Creation and Reproduction of Caricature Style via StableDiffusion, Detection of Visual Artifacts (Evaluation of Generation Quality).
# l'Entraînement d'un Modèle Stable Diffusion avec train_text_to_image
Ce README fournit des instructions détaillées sur la manière d'entraîner un modèle Stable Diffusion en utilisant le script train_text_to_image de la bibliothèque Hugging Face Diffusers. La commande ci-dessous montre comment utiliser divers paramètres pour un entraînement efficace.
## Prérequis
Avant de commencer, assurez-vous d'avoir les prérequis suivants :

- Python installé (version 3.8 ou supérieure recommandée)
- Un environnement configuré avec les packages nécessaires (pensez à utiliser des environnements virtuels)
- Accès à un GPU pour l'entraînement

## Configuration de l'Environnement

### Cloner le Répertoire Diffusers

Clonez le répertoire Hugging Face Diffusers sur votre machine locale :
```sh
git clone https://github.com/huggingface/diffusers.git
cd diffusers
```

### Cloner le Répertoire Diffusers
Installez les packages requis en exécutant :
```sh
pip install -r requirements.txt
pip install accelerate wandb
```
### Préparer votre Jeu de Données
Assurez-vous que votre jeu de données est préparé et accessible. Dans cet exemple, nous utilisons un jeu de données nommé "Acadys/PointConImagesV2".

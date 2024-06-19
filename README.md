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

## Entraînement du Modèle
Utilisez la commande suivante pour démarrer l'entraînement du modèle Stable Diffusion (apres configuration de accelerate et wandb):
```sh
accelerate launch --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --dataset_name="Acadys/PointConImagesV2" \
  --use_ema \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=8 \
  --gradient_checkpointing \
  --max_train_steps=600 \
  --learning_rate=1e-04 \
  --max_grad_norm=1.2 \
  --caption_column=texte \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="Acadys/PointConImageModelV1-4" \
  --push_to_hub \
  --validation_prompts="Un patron en costume donne un dossier à un employé" \
  --report_to="wandb"
```
## Explication des Paramètres de la Commande

- `--mixed_precision="fp16"` : Utilise l'entraînement en précision mixte pour réduire l'utilisation de la mémoire et accélérer les calculs.
- `--pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"` : Chemin vers le modèle Stable Diffusion pré-entraîné.
- `--dataset_name="Acadys/PointConImagesV2"` : Nom du jeu de données à utiliser pour l'entraînement.
- `--use_ema` : Active l'utilisation de la Moyenne Exponentielle Mobile pour les paramètres du modèle afin d'améliorer la stabilité.
- `--resolution=512` : Définit la résolution des images d'entrée.
- `--train_batch_size=4` : Définit la taille du lot pour l'entraînement.
- `--gradient_accumulation_steps=8` : Nombre de pas d'accumulation de gradient pour simuler une taille de lot plus grande.
- `--gradient_checkpointing` : Active le gradient checkpointing pour économiser de la mémoire.
- `--max_train_steps=600` : Définit le nombre maximal de pas d'entraînement.
- `--learning_rate=1e-04` : Taux d'apprentissage pour l'optimiseur.
- `--max_grad_norm=1.2` : Norme maximale de gradient pour le clipping de gradient.
- `--caption_column=texte` : Nom de la colonne dans le jeu de données contenant les légendes.
- `--lr_scheduler="constant"` : Type de planificateur de taux d'apprentissage.
- `--lr_warmup_steps=0` : Nombre de pas pour le warmup du taux d'apprentissage.
- `--output_dir="Acadys/PointConImageModelV1-4"` : Répertoire pour sauvegarder le modèle entraîné.
- `--push_to_hub` : Pousse le modèle vers Hugging Face Hub après l'entraînement.
- `--validation_prompts="Un patron en costume donne un dossier à un employé"` : Invitations pour la validation pendant l'entraînement.
- `--report_to="wandb"` : Rapporte les métriques d'entraînement à Weights & Biases.

## Suivi de l'Entraînement

L'entraînement peut être suivi en utilisant Weights & Biases. Assurez-vous d'avoir configuré votre compte et d'être connecté avant de commencer l'entraînement. Les journaux, métriques et points de contrôle du modèle seront suivis et visualisés dans votre tableau de bord W&B.

## Sauvegarde et Publication du Modèle

Après l'entraînement, le modèle sera sauvegardé dans le répertoire de sortie spécifié et éventuellement poussé vers Hugging Face Hub pour un accès et un partage faciles.


Pour plus d'informations détaillées, consultez la documentation de Hugging Face Diffusers.

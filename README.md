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

# Création et Reproduction de Style Caricatural via Stable Diffusion et Détection des Artefacts Visuels (Évaluation de la Qualité de Génération)

Cette section explique comment créer et reproduire un style caricatural en utilisant Stable Diffusion et comment évaluer la qualité de génération en détectant les artefacts visuels.

## Installation

### Installer les Bibliothèques Nécessaires

Assurez-vous d'avoir installé les bibliothèques nécessaires.

### Importation des Bibliothèques

Nous commençons par importer les bibliothèques nécessaires pour notre projet. Ces bibliothèques incluent des outils pour la manipulation des données, la création de modèles de détection d'objets et la visualisation des résultats.

### Définition de la Classe `ArtifactDataset`

Nous définissons une classe `ArtifactDataset` qui hérite de `torch.utils.data.Dataset`. Cette classe est utilisée pour charger et prétraiter les images et annotations.

#### Initialisation de la Classe

Le constructeur `__init__` initialise les variables et traite les annotations pour extraire les boîtes englobantes et les classes des objets dans les images.

#### Taille du Dataset

La méthode `__len__` renvoie le nombre total d'images dans le dataset.

#### Récupération d'un Échantillon

La méthode `__getitem__` charge une image et ses annotations correspondantes. Elle applique également des transformations aux images et aux boîtes englobantes si elles sont définies.

#### Transformation des Images et des Boîtes

La méthode `transform_image_and_boxes` redimensionne les images et ajuste les boîtes englobantes en conséquence.

### Chemins des Fichiers

Nous définissons les chemins des fichiers pour les images, les annotations et le modèle. Les chemins peuvent être ajustés pour un environnement Google Drive ou un environnement local.

### Affichage des Images avec Boîtes Englobantes

La fonction `display_image_with_boxes` permet d'afficher une image avec ses boîtes englobantes pour visualiser les artefacts détectés. Elle utilise Matplotlib pour tracer les boîtes sur l'image.

### Chargement des Annotations

Nous chargeons les annotations à partir du fichier JSON spécifié. Ces annotations contiennent les informations nécessaires pour chaque image, y compris les coordonnées des boîtes englobantes et les étiquettes des classes.

### Séparation des Données

Nous divisons les chemins des images en ensembles d'entraînement, de validation et de test. Cette séparation permet de former, valider et tester le modèle de manière appropriée.

### Filtrage des Annotations

Les annotations sont filtrées en fonction des ensembles de données d'entraînement, de validation et de test. Cela garantit que chaque ensemble contient les annotations correctes pour les images qu'il contient.

### Transformation des Images

Nous définissons des transformations pour redimensionner les images à une taille fixe et les convertir en tenseurs. Ces transformations sont appliquées lors du chargement des données.

### Création des Datasets

Nous créons les datasets pour l'entraînement, la validation et le test en utilisant les annotations filtrées et les transformations définies.

### Chargement des Données

Nous utilisons `DataLoader` pour charger les données des datasets en lots. Cela permet un traitement efficace des données pendant l'entraînement et l'évaluation.

### Définition du Modèle de Détection d'Objets

La fonction `get_object_detection_model` définit et renvoie un modèle de détection d'objets basé sur Faster R-CNN avec un backbone ResNet-50. Nous modifions le prédicteur pour correspondre au nombre de classes de notre dataset et utilisons une perte de régression de boîte basée sur le Generalized IoU.

### Configuration du Dispositif

Nous déterminons si un GPU est disponible et définissons le modèle pour qu'il utilise le GPU si possible. Cela permet d'accélérer les calculs pendant l'entraînement et l'évaluation du modèle.

### Configuration de l'Optimiseur et du Scheduler

Nous configurons l'optimiseur AdamW pour mettre à jour les poids du modèle avec une certaine valeur de learning rate et de weight decay. Un scheduler est également configuré pour réduire le learning rate lorsque la performance du modèle stagne.

### Définition de la Fonction pour le Remplissage des Boîtes

La fonction `pad_boxes` est utilisée pour compléter les listes de boîtes englobantes avec des boîtes factices pour atteindre une taille maximale spécifiée. Cela est utile pour la comparaison entre les prédictions et les boîtes réelles.

### Boucle d'Entraînement

Nous entrons dans la boucle d'entraînement pour un certain nombre d'époques. Pendant chaque époque, le modèle est entraîné sur les lots d'images et de cibles :

- Le modèle calcule les pertes pour chaque lot, et ces pertes sont ensuite cumulées pour obtenir une perte totale d'entraînement.
- Les gradients sont calculés et les paramètres du modèle sont mis à jour en conséquence.
- Des informations sur la progression de l'entraînement sont affichées périodiquement.

### Évaluation du Modèle

Après chaque époque d'entraînement, le modèle est évalué sur le jeu de validation :

- Le modèle passe en mode évaluation pour désactiver le calcul des gradients.
- Les prédictions du modèle sont obtenues pour chaque image.
- Les boîtes prédictives sont filtrées en fonction d'un seuil de confiance et sont soumises à une suppression non maximale (NMS) pour éliminer les chevauchements excessifs.
- Les boîtes prédictives et les boîtes cibles sont complétées pour être de la même taille, puis une perte de boîte généralisée (Generalized IoU) est calculée.
- La perte moyenne de validation est calculée et affichée.

### Mise à Jour du Learning Rate

Le scheduler ajuste le learning rate en fonction de la performance du modèle sur le jeu de validation.

### Enregistrement du Modèle

À la fin de l'entraînement, les poids du modèle sont sauvegardés dans le fichier spécifié. Cela permet de réutiliser le modèle entraîné sans avoir à le réentraîner à partir de zéro.

### Affichage des Images avec Boîtes Englobantes

La fonction `display_image_with_boxes` permet d'afficher une image avec ses boîtes englobantes réelles et prédites. Cette fonction utilise Matplotlib pour tracer les boîtes sur l'image.

#### Processus de la Fonction:

- **Affichage de l'Image** : L'image est affichée en utilisant Matplotlib.
- **Boîtes Réelles** : Les boîtes englobantes réelles (true_boxes) sont tracées en vert.
- **Suppression Non Maximale (NMS)** : Les boîtes prédites (pred_boxes) et les scores associés (pred_scores) sont filtrées à l'aide de la suppression non maximale pour éliminer les chevauchements excessifs.
- **Boîtes Prédites** : Les boîtes englobantes prédites, qui ont un score supérieur ou égal à un seuil de confiance spécifié, sont tracées en rouge.

### Affichage des Résultats

Pour chaque image du lot, les boîtes réelles et prédites sont extraites et passées à la fonction `display_image_with_boxes` pour être affichées avec leurs annotations respectives.

### Définition des Seuils

Nous définissons des seuils pour la confiance (confidence_threshold), l'IoU (iou_threshold) et le taux de contenance (containment_threshold). Ces seuils déterminent quelles boîtes prédictives sont considérées comme positives.

### Calcul des Métriques

#### Précision et Rappel

Les fonctions de calcul de précision et de rappel déterminent respectivement la proportion de prédictions correctes parmi les prédictions totales et la proportion de véritables positifs détectés parmi les positifs réels.

- **Précision** : \( \text{Précision} = \frac{TP}{TP + FP} \)
- **Rappel** : \( \text{Rappel} = \frac{TP}{TP + FN} \)

#### F1 Score

Le F1 score est la moyenne harmonique de la précision et du rappel, offrant un équilibre entre les deux.

- **F1 Score** : \( F1 = 2 \times \frac{\text{Précision} \times \text{Rappel}}{\text{Précision} + \text{Rappel}} \)

#### Exactitude

L'exactitude est le rapport entre les véritables positifs et la somme des véritables positifs, faux positifs et faux négatifs.

- **Exactitude** : \( \text{Exactitude} = \frac{TP}{TP + FP + FN} \)

#### Contenance

La fonction de contenance vérifie si une boîte prédictive est principalement contenue dans une boîte réelle, selon un seuil de contenance spécifié.

- **Contenance** : \( \text{Contenance} = \frac{\text{aire de l’intersection}}{\text{aire de la boîte réelle}} \)

### Calcul des Métriques sur les Boîtes

La fonction `calculate_metrics` calcule les métriques de performance des boîtes prédictives par rapport aux boîtes réelles. Elle retourne la précision, le rappel, le F1 score, l'exactitude, l'IoU moyen, et des détails sur chaque prédiction.

### Évaluation du Modèle sur l'Ensemble de Test

La fonction `evaluate_model_on_test_set` évalue le modèle sur l'ensemble de test. Elle calcule les métriques globales de performance (précision, rappel, F1 score, exactitude, IoU moyen) en agrégeant les résultats de chaque image du test.

#### Processus d'Évaluation:

- **Mode Évaluation** : Le modèle est mis en mode évaluation.
- **Prédictions et Filtrage** : Les prédictions sont obtenues pour chaque image et filtrées en fonction du seuil de confiance. Les prédictions sont également soumises à la suppression non maximale (NMS) pour éliminer les boîtes redondantes.
- **Calcul des Métriques** : Les métriques sont calculées pour chaque paire d'image et de boîtes réelles/prédictives.
- **Agrégation des Résultats** : Les métriques globales sont calculées en agrégeant les résultats de toutes les images.

### Affichage des Résultats

Les résultats des métriques globales (précision, rappel, F1 score, exactitude, IoU moyen) sont affichés, offrant une vue d'ensemble des performances du modèle sur l'ensemble de test. Les détails des prédictions pour chaque image sont également imprimés pour une analyse plus approfondie.

### Valeurs des Métriques sur l'Ensemble de Test des 284 Générations Initiales

- **Précision**: 0.9341726697041658 (93.42%)
- **Rappel**: 0.8852939540200799 (88.53%)
- **F1 Score**: 0.9090767653732305 (90.91%)
- **Exactitude**: 0.8333095643381793 (83.33%)
- **IoU Moyen**: 0.8502013724366815 (85.02%)

## Metrics

### Définition des Seuils

Les seuils sont définis pour la confiance (confidence_threshold), l'IoU (iou_threshold) et le taux de contenance (containment_threshold). Ces seuils déterminent quelles boîtes prédictives sont considérées comme positives.

### Lecture de l'Image

La fonction `read_image` lit une image à partir d'un chemin spécifié et applique des transformations pour la convertir en tenseur.

### Détection sur une Image

La fonction `detect_image` effectue la détection d'objets sur une image donnée en utilisant un modèle de détection et renvoie les boîtes prédictives et les scores associés. Les boîtes prédictives sont filtrées en fonction du seuil de confiance et soumises à la suppression non maximale (NMS).

### Affichage des Images avec Boîtes Englobantes

La fonction `display_image_with_boxes` affiche une image avec les boîtes englobantes réelles et prédites en utilisant Matplotlib. Les boîtes réelles sont tracées en vert et les boîtes prédictives en rouge avec leurs scores de confiance.

### Vérification de la Contenance

La fonction `is_contained` vérifie si une boîte prédictive est principalement contenue dans une boîte réelle en utilisant un seuil de contenance spécifié.

### Calcul des Métriques

La fonction `calculate_metrics` calcule les métriques de performance des boîtes prédictives par rapport aux boîtes réelles, telles que la précision, le rappel, le F1 score, l'exactitude et l'IoU moyen. Elle retourne également des détails sur chaque prédiction.

## Traitement de Toutes les Images

La fonction `process_all_images` traite toutes les images d'un dossier spécifié, lit les annotations réelles, effectue la détection d'objets, calcule les métriques de performance et affiche les images avec les boîtes englobantes réelles et prédites.

### Processus:

- **Chargement des Annotations** : Les annotations réelles sont chargées à partir d'un fichier JSON.
- **Détection et Affichage** : Pour chaque image, les boîtes prédictives sont obtenues et affichées avec les boîtes réelles.
- **Calcul des Métriques Globales** : Les métriques globales sont calculées en agrégeant les résultats de toutes les images.

### Affichage des Résultats

Les résultats des métriques pour chaque image sont affichés, ainsi que les métriques globales sur toutes les images de test. Cela inclut la précision globale, le rappel, le F1 score, l'exactitude et l'IoU moyen.

## Métriques Globales sur les 24 Images d'Éléments Totalement Nouveaux de la Deuxième Vague d'Images Générées

- **Précision Globale**: 0.91
- **Rappel Global**: 0.71
- **F1 Score Global**: 0.80
- **Exactitude Globale**: 0.67
- **IoU Moyen Global**: 0.67


# Learning Visual Representations — Vue d'ensemble

## Objectif du TD

Ce TD porte sur l'**apprentissage auto-supervisé de représentations visuelles**. Il reproduit des résultats de deux articles fondateurs :

1. **Deep Clustering (DC)** — Caron et al. (2018) : apprendre des features visuelles non supervisées par clustering itératif.
2. **SimCLR** — Chen et al. (2020) : apprentissage contrastif en maximisant la similarité entre deux vues augmentées d'une même image.

Le modèle de base est **AlexNet**, pré-entraîné par Deep Clustering sur ImageNet. On évalue ensuite la qualité des représentations apprises via classification (linéaire ou KNN).

---

## Arborescence des fichiers

```
learning_visual_representations/
├── configs/                          # Fichiers de configuration YAML
│   ├── dc_exp.yaml                   # Config classification downstream (DC)
│   ├── contrastive_training.yaml     # Config entraînement SimCLR standard
│   ├── contrastive_training_background.yaml
│   ├── contrastive_training_digit.yaml
│   ├── contrastive_training_background_digit.yaml
│   ├── clf_background.yaml           # Config évaluation KNN (background)
│   ├── clf_digit.yaml                # Config évaluation KNN (digit)
│   ├── clf_background_digit.yaml     # Config évaluation KNN (background+digit)
│   └── hyperparameter_search.yaml    # Config recherche d'hyperparamètres
│
├── learning/
│   ├── transformations.py            # Classe DataTransformation (augmentations)
│   └── nce_loss.py                   # Loss NT-Xent (SimCLR)
│
├── models/
│   ├── alexnet.py                    # Architecture AlexNet + Sobel + freeze
│   └── projection_head.py           # Tête de projection SimCLR (MLP)
│
├── util_scripts/
│   ├── build_imagenet_mnist_test_dataset.py  # Construction dataset test ImageNet+MNIST
│   └── random_search_contrastive.py          # Recherche hyperparamètres (Optuna)
│
├── figures/                          # Images pour le notebook
│
├── activations.py                    # Extraction d'activations + gradient ascent
├── contrastive_training.py           # Script d'entraînement contrastif SimCLR
├── datasets.py                       # Datasets : ImageNet, ContrastiveDataset, ImageNetMnist
├── dc_exp.py                         # Expérience classification downstream (DC)
├── imagenet_mnist_exp.py             # Évaluation KNN sur ImageNet+MNIST
├── imagenet_mnist_training.py        # Entraînement contrastif sur ImageNet+MNIST
├── utils.py                          # Utilitaires (lecture image, stats, visualisation)
└── learning_visual_representations.ipynb  # Notebook principal du TD
```

---

## Description détaillée de chaque fichier

### `datasets.py` — Jeux de données

| Classe / Fonction | Description | À compléter ? |
|---|---|---|
| `ImageNet(Dataset)` | Dataset de base : lit des images depuis un dossier + fichier de labels. | ✅ `__getitem__` : conversion tensor, permutation axes, conversion float (`...`) |
| `SubsetImageNet(ImageNet)` | Filtre ImageNet pour ne garder que certaines classes. | ✅ Complet |
| `ContrastiveDataset(SubsetImageNet)` | Retourne `(img, view1, view2)` — deux vues augmentées de la même image pour l'apprentissage contrastif. | ✅ `img1 = ...`, `img2 = ...` |
| `ImageNetMnist(SubsetImageNet)` | Superpose des chiffres MNIST sur des images ImageNet. Supporte 3 modes : `'background'`, `'digit'`, `['background', 'digit']`. | ✅ Le corps de chaque branche `shared_feature` contient `...` |
| `insert_digit(digit, img)` | Incruste un chiffre MNIST sur une image. | ✅ Complet |
| `collate_views(batch)` | Fonction de collation pour `ImageNetMnist`. | ✅ Complet |

### `models/alexnet.py` — AlexNet

| Élément | Description | À compléter ? |
|---|---|---|
| `AlexNet(nn.Module)` | AlexNet avec filtre Sobel optionnel et gel des couches conv. | ✅ Complet |
| `freeze_feature_layers()` | Gèle les poids des couches convolutionnelles. Docstring "TODO" mais le code est implémenté. | ⚠️ Vérifier la compréhension |
| `alexnet(...)` | Factory function pour créer un AlexNet. | ✅ Complet |

### `models/projection_head.py` — Tête de projection SimCLR

| Élément | Description | À compléter ? |
|---|---|---|
| `ProjectionHead(nn.Module)` | MLP de projection (encoder → espace contrastif). Doit être `Linear → ReLU → Linear` (d_in → d_in → d_model). | ✅ **Entièrement à implémenter** : `__init__` vide, `forward` → `NotImplementedError` |

### `learning/transformations.py` — Augmentations de données

| Élément | Description | À compléter ? |
|---|---|---|
| `DataTransformation` | Classe factory qui construit un pipeline de transformations à partir d'un dict de config. Supporte : `center_cropping`, `random_cropping`, `resize`, `color_distortion`, `gaussian_blur`, `normalize`. | ✅ **Les 6 transformations sont `...`** (à remplacer par les appels `T.CenterCrop`, `T.RandomCrop`, `T.Resize`, `T.ColorJitter`, `T.GaussianBlur`, `T.Normalize`) |

### `learning/nce_loss.py` — Loss NT-Xent

| Élément | Description | À compléter ? |
|---|---|---|
| `nce_loss(z1, z2, temperature)` | Implémentation de la loss NT-Xent de SimCLR. Doit : normaliser les embeddings, calculer la matrice de similarité cosinus, masquer la diagonale, appliquer le scaling par température, puis `F.cross_entropy`. | ✅ **Corps entièrement manquant** entre les constantes et `loss = F.cross_entropy(sim, targets)` |

### `activations.py` — Visualisation des filtres

| Fonction | Description | À compléter ? |
|---|---|---|
| `compute_batch_activations(model, x, layer)` | Extrait les activations d'une couche ReLU donnée. | ✅ Forward pass `x = ...`, capture `activation = ...` |
| `compute_activations_for_gradient_ascent(model, x, layer, filter_id)` | Idem mais pour le gradient ascent (capture activation Conv2d). | ✅ Plusieurs `...` |
| `compute_dataset_activations(model, dataset, layer)` | Boucle sur le dataset, appelle `compute_batch_activations`. | ✅ Complet (dépend du précédent) |
| `maximize_img_response(...)` | Gradient ascent dans l'espace image pour maximiser la réponse d'un filtre. Régularisation L2 + blur gaussien. | ✅ `target = ...`, `loss = ...`, mise à jour `...` |

### `contrastive_training.py` — Entraînement SimCLR (standard)

| Élément | Description | À compléter ? |
|---|---|---|
| `main(cfg)` | Pipeline complet : charge dataset contrastif, AlexNet pré-entraîné + ProjectionHead, entraîne avec NCE loss, sauvegarde le meilleur modèle. | ✅ `optimizer = optim.Adam(...)` (params `...`), boucle : `h1, h2 = ...`, `z1, z2 = ...`, `loss = ...` |

### `imagenet_mnist_training.py` — Entraînement contrastif ImageNet+MNIST

| Élément | Description | À compléter ? |
|---|---|---|
| `main(cfg)` | Similaire à `contrastive_training.py` mais sur le dataset ImageNet+MNIST avec des vues spécifiques selon `shared_feature`. | ✅ Boucle d'entraînement : `h1, h2 = ...`, `z1, z2 = ...`, `loss = ...` |

### `dc_exp.py` — Classification downstream (Deep Clustering)

| Élément | Description | À compléter ? |
|---|---|---|
| `main(cfg)` | Charge AlexNet avec features gelées, entraîne un classifieur linéaire, évalue sur train/val/test. | ✅ `logits = ...`, `loss = ...`, `pred = ...`, `accuracy = ...` (train + val) + `...` final (test) |

### `imagenet_mnist_exp.py` — Évaluation KNN

| Élément | Description | À compléter ? |
|---|---|---|
| `main(cfg, task)` | Charge un AlexNet contrastif, extrait les features, entraîne un classifieur KNN, retourne l'accuracy. | ✅ Complet |

### `utils.py` — Utilitaires

| Fonction | Description | À compléter ? |
|---|---|---|
| `read_image(path)` | Lecture d'image avec jpeg4py (fallback OpenCV). | ✅ Complet |
| `show_samples_per_class(...)` | Affiche N échantillons d'une classe. | ✅ Complet |
| `data_stats(dataset)` | Calcule mean/std par canal. | ✅ Complet |
| `dataset2tensor(dataset)` | Convertit un dataset en tenseur. | ✅ Complet |
| `deprocess_image(x, mean, std)` | Dé-normalise une image pour visualisation. | ✅ Complet |
| `MidpointNormalize` | Normalisation matplotlib avec midpoint. | ✅ Complet |

### Fichiers de config YAML (`configs/`)

| Fichier | Rôle | Paths renseignés ? |
|---|---|---|
| `dc_exp.yaml` | Classification downstream DC | ✅ Chemins remplis |
| `contrastive_training.yaml` | Entraînement SimCLR standard | ❌ Chemins vides |
| `contrastive_training_background.yaml` | SimCLR vues "background" partagé | ❌ Chemins vides |
| `contrastive_training_digit.yaml` | SimCLR vues "digit" partagé | ❌ Chemins vides |
| `contrastive_training_background_digit.yaml` | SimCLR vues "background+digit" partagés | ❌ Chemins vides |
| `clf_background.yaml` | KNN évaluation (background) | ❌ Chemins vides |
| `clf_digit.yaml` | KNN évaluation (digit) | ❌ Chemins vides |
| `clf_background_digit.yaml` | KNN évaluation (background+digit) | ❌ Chemins vides |
| `hyperparameter_search.yaml` | Recherche Optuna | ❌ Chemins vides |

---

## Structure du notebook — Déroulement logique

Le notebook suit le plan suivant :

### Partie 1 — Deep Clustering

| Étape | Cellules | Ce qui est demandé |
|---|---|---|
| **Setup** | 1–8 | Import, chargement config + dataset, visualisation des images. |
| **1.1 Prior des conv** | 9–16 | Charger un AlexNet random avec features gelées → entraîner un classifieur linéaire → vérifier que c'est mieux que le hasard. Nécessite de compléter `dc_exp.py`. |
| **1.2.1 Filtres couche 1** | 17–22 | Charger AlexNet pré-entraîné DC, visualiser les filtres conv1. |
| **1.2.2 Couches profondes** | 23–33 | Compléter `activations.py` : extraire les activations de la couche 5, trouver les 10 images qui activent le plus les filtres 0 et 33 (`top10_activations_f0/f33 = ...`). Puis gradient ascent pour maximiser la réponse d'un filtre. |

### Partie 2 — Contrastive Learning (SimCLR)

| Étape | Cellules | Ce qui est demandé |
|---|---|---|
| **Setup contrastif** | 34–38 | Import, chargement config contrastive, visualisation des paires de vues augmentées. Nécessite `DataTransformation` et `ContrastiveDataset` complétés. |
| **Similarité DC** | 39–41 | Calculer la similarité cosinus entre vues avec le modèle DC (`sim = ...`). |
| **2.1 Implémentation SimCLR** | 42–49 | Compléter `nce_loss.py`, `projection_head.py`, `contrastive_training.py`. Entraîner SimCLR. Calculer la similarité cosinus après entraînement (`model = ...`, `contrastive_sim = ...`). Comparer avec DC. |
| **2.2 Vues optimales** | 50–56 | Reproduire Table 2 de Tian et al. : compléter `ImageNetMnist`, `imagenet_mnist_training.py`, `imagenet_mnist_exp.py`. Entraîner 3 modèles (shared = background / digit / les deux) et évaluer par KNN. Remplir le tableau des taux d'erreur. |

---

## Résumé des TODOs — Ordre recommandé

Voici la liste de tout ce qui est à compléter, dans un ordre logique :

### Étape 1 : Fondations (datasets + utils)
1. **`datasets.py` → `ImageNet.__getitem__`** : conversion tensor, permutation axes, float.
2. **`datasets.py` → `ContrastiveDataset.__getitem__`** : appliquer `self.view_transform` pour créer `img1` et `img2`.

### Étape 2 : Deep Clustering (partie 1 du notebook)
3. **`dc_exp.py` → boucle train** : `logits`, `loss`, `pred`, `accuracy`.
4. **`dc_exp.py` → boucle val** : idem.
5. **`dc_exp.py` → test** : évaluation finale.
6. **`activations.py` → `compute_batch_activations`** : forward pass `x = m(x)`, capture de l'activation.
7. **`activations.py` → `compute_activations_for_gradient_ascent`** : idem avec filtre spécifique.
8. **`activations.py` → `maximize_img_response`** : target, loss, gradient, mise à jour image.
9. **Notebook** : `top10_activations_f0 = ...` et `top10_activations_f33 = ...`.

### Étape 3 : SimCLR (partie 2 du notebook)
10. **`learning/transformations.py` → `DataTransformation.__init__`** : les 6 transformations.
11. **`models/projection_head.py` → `ProjectionHead`** : `__init__` (couches MLP) + `forward`.
12. **`learning/nce_loss.py` → `nce_loss`** : calculer la matrice de similarité, les targets, appliquer le masque diagonal.
13. **`contrastive_training.py` → `main`** : paramètres de l'optimiseur, boucle d'entraînement (`h1/h2`, `z1/z2`, `loss`).
14. **Notebook** : `sim = ...` (similarité DC), `model = ...` (chargement modèle SimCLR), `contrastive_sim = ...`.

### Étape 4 : ImageNet+MNIST (partie 2.2 du notebook)
15. **`datasets.py` → `ImageNetMnist.__getitem__`** : les 3 branches `shared_feature`.
16. **`imagenet_mnist_training.py` → boucle train** : `h1/h2`, `z1/z2`, `loss`.
17. **Remplir les configs YAML** avec les bons chemins de données.
18. **Entraîner les 3 modèles** (background / digit / background+digit) et évaluer par KNN.
19. **Remplir le tableau** des taux d'erreur dans le notebook.

### Optionnel
20. **`util_scripts/random_search_contrastive.py`** : bornes de recherche + boucle train.
21. **`util_scripts/build_imagenet_mnist_test_dataset.py`** : corriger les chemins hardcodés.

---

## Notes et pièges potentiels

- **Filtre Sobel** : le modèle DC pré-entraîné utilise `sobel=True` (entrée 2 canaux au lieu de 3 RGB). Penser à bien mettre `sobel=True` lors du chargement.
- **`d_alexnet`** : le modèle DC pré-entraîné a une sortie de 10000 dimensions pour `top_layer`. Le `d_alexnet` dans les configs contrastifs doit correspondre.
- **Config `hyperparameter_search.yaml`** : utilise `'cropping'` au lieu de `'random_cropping'` ou `'center_cropping'` — possible bug de compatibilité avec `DataTransformation`.
- **`build_imagenet_mnist_test_dataset.py`** : contient des chemins hardcodés type `'path/to/folder'` et un double `configs/configs/` — à corriger.
- **GPU** : le gradient ascent (`maximize_img_response`) et l'entraînement SimCLR bénéficient grandement du GPU.

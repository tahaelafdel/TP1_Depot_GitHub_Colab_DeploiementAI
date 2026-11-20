# TP1 — Déploiement de ResNet-18 sur CPU et GPU

Le projet compare quatre chaînes d’inférence Google Colab sur les mêmes 50 images de test CIFAR-10, avec un lot de 10 et des calculs FP32. Le backbone provient des poids ImageNet de ResNet-18. Une tête linéaire entraînée sur le split d’entraînement de CIFAR-10 produit les dix classes du TP.

## Notebooks

| Fichier | Chaîne | Runtime Colab |
| --- | --- | --- |
| [03_GPU_TensorRT.ipynb](notebooks/03_GPU_TensorRT.ipynb) | Préparation commune puis PyTorch → ONNX → TensorRT → CUDA | GPU NVIDIA |
| [04_GPU_TorchScript.ipynb](notebooks/04_GPU_TorchScript.ipynb) | PyTorch → TorchScript → CUDA, puis synthèse | GPU NVIDIA |
| [01_CPU_OpenVINO.ipynb](notebooks/01_CPU_OpenVINO.ipynb) | PyTorch → ONNX → OpenVINO → CPU | CPU |
| [02_CPU_TorchScript.ipynb](notebooks/02_CPU_TorchScript.ipynb) | PyTorch → TorchScript → CPU | CPU |

## Exécution sur Colab

1. Importer les quatre fichiers `.ipynb` via **Fichier → Importer un notebook**. Utiliser le même compte Google pour le dossier Drive `MyDrive/TP1_ResNet18`. Sur Colab, choisir l’accélérateur dans **Exécution → Modifier le type d’exécution**.
2. Exécuter **03** sur GPU en premier. Il prépare et sauvegarde les poids, l’export ONNX et les 50 images normalisées, puis mesure TensorRT. La préparation est réutilisée tant que son manifeste est présent.
3. Exécuter **01** et **02** en mode CPU, puis **04** sur GPU. Relever les modèles matériels affichés. Utiliser le même type de GPU pour **03** et **04** si Colab le permet. Garder le même nom de campagne dans les quatre notebooks et choisir un nouveau nom pour une campagne distincte.
4. Dans **04**, exécuter la section **Synthèse et rapport**. Elle valide les 12 fichiers JSON, affiche les graphiques et génère `TP1_Resultats_colab-tp1.zip` avec le nom de campagne par défaut sur Drive. La campagne jointe à ce dépôt est `colab-tp1`.
5. Publier le dépôt en visibilité **Public**, vérifier l’accès en navigation privée, puis déposer son URL sur Moodle.

## Fichiers livrés

- `notebooks/` contient les quatre implémentations et le code de synthèse.
- [Rapport_TP1_Colab.pdf](Rapport_TP1_Colab.pdf) présente l’architecture, le protocole et le bilan des 12 exécutions Colab.
- `resultats/` contient les 12 JSON bruts, le manifeste SHA-256, les tableaux CSV et cinq graphiques. `accord_predictions.csv` détaille les 66 comparaisons par paires.

Chaque pipeline effectue trois lancements dans des processus distincts. Après échauffement, chaque lancement mesure dix passages sur le même sous-ensemble. La latence couvre les lots prétraités en RAM jusqu’aux logits en RAM, y compris les transferts GPU. Les valeurs énergétiques dépendent de la disponibilité des compteurs NVML ou RAPL et sont exprimées en joules et joules par image.

## Principales conclusions

- **CPU** · OpenVINO atteint 17,04 images/s contre 11,76 pour TorchScript, soit environ 45 % de débit en plus.
- **Tesla T4** · TensorRT atteint 1 150,84 images/s contre 816,62 pour TorchScript, soit environ 41 % de débit en plus.
- **Prédictions** · 76 % Top-1 dans chaque exécution, soit 38 images correctement classées sur 50. Les 12 exécutions prédisent la même classe sur chacune des 50 images.
- **Énergie GPU** · TensorRT utilise en moyenne 68,8 mJ/image contre 78,9 mJ/image pour TorchScript. Les mesures d’énergie CPU sont indisponibles.
- **Variabilité** · L’écart-type relatif du débit atteint au maximum 4,4 % pour TorchScript sur CPU.

# Evaluation Grid — Image & Text (real)

Scope: **two modalities — image (FashionMNIST) and text (IMDB)**. The tabular (Credit-card Fraud) case study has been removed. From `experiments/real_full/summary.csv`. Cell = **BSR / AUROC** (`–` = n/a).

> ⚠️ The IMDB rows below are the **old from-scratch TextEmbedMLP (underfit, acc≈0.5–0.66)**. The text model has since been switched to **DistilBERT** (frozen encoder + head; fixes underfit, acc≈0.8+); rerun the IMDB grid with `trustfl/configs/imdb_ecf.yaml` (now `model: distilbert`) to refresh them.


## FashionMNIST (image, 60 rounds, root=100)

| attack | fedavg | median | trimmed_mean | multi_krum | fltrust | ecf_base | ecf_cand | ecf_bdoor |
|---|---|---|---|---|---|---|---|---|
| label_flip | –/– | –/– | –/– | –/1.00 | –/1.00 | –/1.00 | –/1.00 | –/1.00 |
| backdoor | 1.00/– | 0.15/– | 0.39/– | 0.01/1.00 | 0.04/1.00 | 0.17/0.92 | 0.08/0.86 | 0.17/0.20 |
| spurious_feature | –/– | –/– | –/– | –/0.84 | –/0.53 | –/0.81 | –/0.61 | –/0.44 |
| constrained_backdoor | 1.00/– | 0.14/– | 0.32/– | 0.01/1.00 | 0.04/1.00 | 0.16/0.94 | 0.08/0.84 | 0.17/0.48 |
| adaptive_ecf | 0.98/– | 0.12/– | 0.23/– | 0.66/0.09 | 0.04/1.00 | 0.16/0.94 | 0.09/0.86 | 0.17/0.31 |
| sign_flip | –/– | –/– | –/– | –/0.95 | –/1.00 | –/0.97 | –/0.95 | –/1.00 |
| gaussian | –/– | –/– | –/– | –/1.00 | –/1.00 | –/0.39 | –/0.05 | –/0.92 |
| lie | –/– | –/– | –/– | –/0.81 | –/1.00 | –/1.00 | –/0.94 | –/1.00 |
| min_max | –/– | –/– | –/– | –/1.00 | –/1.00 | –/1.00 | –/0.94 | –/1.00 |

## IMDB (text, 30 rounds) — OLD TextEmbedMLP, to refresh

| attack | fedavg | median | trimmed_mean | multi_krum | fltrust | ecf_base | ecf_cand | ecf_bdoor |
|---|---|---|---|---|---|---|---|---|
| label_flip | –/– | –/– | –/– | –/0.27 | –/0.78 | –/0.16 | –/0.36 | · |
| backdoor | 0.35/– | 0.71/– | 0.46/– | 0.81/0.38 | 0.98/0.84 | 0.93/0.36 | 0.96/0.69 | · |
| spurious_feature | –/– | –/– | –/– | –/0.33 | –/0.38 | –/0.70 | –/0.34 | · |
| constrained_backdoor | 0.35/– | 0.71/– | 0.46/– | 0.81/0.38 | 0.98/0.84 | 0.93/0.36 | 0.96/0.69 | · |
| adaptive_ecf | 0.75/– | 0.45/– | 0.27/– | 0.19/0.48 | 0.87/0.20 | 0.72/0.05 | 0.75/0.19 | · |
| sign_flip | –/– | –/– | –/– | –/0.42 | –/0.81 | –/0.22 | –/0.44 | · |
| gaussian | –/– | –/– | –/– | –/1.00 | –/0.77 | –/0.00 | –/0.00 | · |
| lie | –/– | –/– | –/– | –/0.75 | –/0.75 | –/0.00 | –/0.00 | · |
| min_max | –/– | –/– | –/– | –/1.00 | –/1.00 | –/1.00 | –/1.00 | · |

*For the root=500 image grid (where ECF's contribution is clearest) see `experiments/fmnist_r500/`; intermittent-attack study in `experiments/intermittent/`.*

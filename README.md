# MSHF-Net

This code is a **pytorch** implementation of our paper "**MSHF-Net: Multimodal Breast Cancer Molecular Subtype Prediction via Segmentation-Guided Hierarchical Fusion Network**"

## Proposed method

![overall architecture](docs/architecture.png)

## Comparisons

All results are reported as **mean ± 95% confidence interval half-width**. Best values are shown in bold, and second-best values are underlined.

### Table 1. Backbone comparison across modalities.

M denotes MG, U denotes US, and C denotes clinical data.

| Method | Modality | Acc (%) | F1 (%) | Rec (%) | Pre (%) | AUC (%) |
|---|---|---:|---:|---:|---:|---:|
| <b>ResNet-50 [5]</b> | M+U+C | <b>82.1±1.7</b> | <b>76.1±2.0</b> | <b>71.8±2.6</b> | <b>80.9±1.7</b> | <b>85.3±3.2</b> |
| Inception-v3 [23] | M+U+C | 77.4±2.2 | 65.6±2.6 | 63.0±2.5 | 68.5±2.6 | 79.5±2.8 |
| <u>DenseNet-121 [7]</u> | M+U+C | <u>80.2±1.9</u> | <u>70.5±2.2</u> | <u>68.0±2.4</u> | <u>73.2±2.0</u> | <u>83.6±2.5</u> |
| ViT-B [3] | M+U+C | 78.9±2.0 | 68.5±2.3 | 65.5±2.1 | 71.8±2.4 | 81.4±2.1 |

#### AUC-ranked best-vs-second-best p-value analysis for Table 1.

P-values compare the AUC-best method with the AUC-second-best method using two-sided paired t-tests on the five-fold metrics.

| Comparison | Acc | F1 | Rec | Pre | AUC |
|---|---:|---:|---:|---:|---:|
| ResNet-50 vs DenseNet-121 | 0.0004 (***) | <0.0001 (***) | 0.0009 (***) | <0.0001 (***) | 0.0033 (**) |

Significance symbols: * p < 0.05, ** p < 0.01, *** p < 0.001, ns not significant.

### Table 2. MSHF-Net compared with multimodal baselines.

M denotes MG, U denotes US, and C denotes clinical data.

| Method | Modality | Acc (%) | F1 (%) | Rec (%) | Pre (%) | AUC (%) |
|---|---|---:|---:|---:|---:|---:|
| Part A: Task-Specific Methods |  |  |  |  |  |  |
| Luo et al. [14] | M | 74.5±2.4 | 58.6±2.6 | 55.4±3.1 | 62.1±2.8 | 70.2±2.5 |
| Ben et al. [1] | M+C | 73.2±2.1 | 61.2±2.5 | 58.2±2.6 | 64.5±2.2 | 73.5±2.3 |
| <u>MDL-IIA [26]</u> | M+U | <u>80.5±1.3</u> | <u>72.7±1.6</u> | <u>69.5±1.8</u> | <u>76.2±1.5</u> | <u>82.8±1.4</u> |
| Part B: Classical Methods |  |  |  |  |  |  |
| MMBT [8] | M+U+C | 76.4±1.8 | 66.5±2.1 | 63.8±2.2 | 69.5±1.9 | 77.8±1.8 |
| CLIP [18] | M+U+C | 75.2±2.2 | 63.8±2.4 | 61.0±2.5 | 66.8±2.3 | 75.5±2.1 |
| Part C: Cross-Domain Methods |  |  |  |  |  |  |
| HFBSurv [12] | M+U+C | 77.5±1.7 | 68.2±1.9 | 65.2±2.0 | 71.4±1.8 | 79.2±1.6 |
| TMI-CLNet [24] | M+U+C | 78.8±1.5 | 70.4±1.8 | 67.4±1.9 | 73.6±1.7 | 80.5±1.5 |
| UniCross [25] | M+U+C | 75.8±1.9 | 65.3±2.2 | 62.5±2.4 | 68.3±2.0 | 76.4±1.9 |
| <b>MSHF-Net (Ours)</b> | M+U+C | <b>82.1±1.7</b> | <b>76.1±2.0</b> | <b>71.8±2.6</b> | <b>80.9±1.7</b> | <b>85.3±3.2</b> |

#### AUC-ranked best-vs-second-best p-value analysis for Table 2.

P-values compare the AUC-best method with the AUC-second-best method using two-sided paired t-tests on the five-fold metrics.

| Comparison | Acc | F1 | Rec | Pre | AUC |
|---|---:|---:|---:|---:|---:|
| MSHF-Net vs MDL-IIA | 0.0027 (**) | <0.0001 (***) | 0.0107 (*) | <0.0001 (***) | 0.0191 (*) |

Significance symbols: * p < 0.05, ** p < 0.01, *** p < 0.001, ns not significant.

### Table 3. H-TMF fusion strategies compared.

| Method | Acc (%) | F1 (%) | Rec (%) | Pre (%) | AUC (%) |
|---|---:|---:|---:|---:|---:|
| Early Fusion [4] | 78.8±1.5 | 70.1±1.4 | <u>72.7±2.8</u> | 69.4±2.2 | 75.6±1.8 |
| Late Fusion [4] | 76.1±1.0 | 71.1±1.2 | 70.7±1.4 | 73.4±3.8 | 76.8±1.4 |
| <u>Attention Fusion [11]</u> | <u>80.0±2.7</u> | <u>73.4±2.1</u> | 72.0±1.8 | <u>77.8±3.0</u> | <u>83.7±3.1</u> |
| <b>H-TMF Module (Ours)</b> | <b>82.1±1.7</b> | <b>76.1±2.0</b> | <b>71.8±2.6</b> | <b>80.9±1.7</b> | <b>85.3±3.2</b> |

#### AUC-ranked best-vs-second-best p-value analysis for Table 3.

P-values compare the AUC-best method with the AUC-second-best method using two-sided paired t-tests on the five-fold metrics.

| Comparison | Acc | F1 | Rec | Pre | AUC |
|---|---:|---:|---:|---:|---:|
| H-TMF Module vs Attention Fusion | 0.0103 (*) | <0.0001 (***) | 0.7149 (ns) | 0.0049 (**) | 0.0001 (***) |

Significance symbols: * p < 0.05, ** p < 0.01, *** p < 0.001, ns not significant.

### Table 4. Input modality ablation study.

| MG | US | Clinical | Acc (%) | F1 (%) | Rec (%) | Pre (%) | AUC (%) |
|---|---|---|---:|---:|---:|---:|---:|
| ✓ |  |  | 69.5±2.4 | 54.2±2.8 | 52.5±2.6 | 61.2±2.7 | 68.5±2.5 |
|  | ✓ |  | 70.8±2.1 | 53.5±2.5 | 52.0±2.4 | 60.5±2.2 | 67.4±2.3 |
| ✓ | ✓ |  | <u>80.2±1.8</u> | <u>74.8±2.0</u> | <u>73.2±1.8</u> | <u>75.5±1.9</u> | <u>81.8±1.7</u> |
|  | ✓ | ✓ | 78.5±1.6 | 67.2±1.8 | 66.5±1.7 | 74.2±1.6 | 80.5±1.5 |
| <b>✓</b> | <b>✓</b> | <b>✓</b> | <b>82.1±1.7</b> | <b>76.1±2.0</b> | <b>71.8±2.6</b> | <b>80.9±1.7</b> | <b>85.3±3.2</b> |

#### AUC-ranked best-vs-second-best p-value analysis for Table 4.

P-values compare the AUC-best modality setting with the AUC-second-best modality setting using two-sided paired t-tests on the five-fold metrics.

| Comparison | Acc | F1 | Rec | Pre | AUC |
|---|---:|---:|---:|---:|---:|
| MG+US+Clinical vs MG+US | 0.0014 (**) | <0.0001 (***) | 0.0516 (ns) | <0.0001 (***) | 0.0032 (**) |

Significance symbols: * p < 0.05, ** p < 0.01, *** p < 0.001, ns not significant.

### Table 5. Component ablation study.

| SGA module | CBF Loss | Acc (%) | F1 (%) | Rec (%) | Pre (%) | AUC (%) |
|---|---|---:|---:|---:|---:|---:|
| x | x | 75.8±2.1 | 63.6±2.9 | 59.5±3.1 | 68.2±2.8 | 77.4±2.5 |
| ✓ | x | 77.5±1.9 | 69.3±2.3 | <u>68.2±2.5</u> | 70.4±2.4 | <u>80.5±2.1</u> |
| x | ✓ | <u>81.6±1.8</u> | <u>69.4±2.2</u> | 62.8±2.7 | <u>77.5±2.1</u> | 78.1±2.3 |
| <b>✓</b> | <b>✓</b> | <b>82.1±1.7</b> | <b>76.1±2.0</b> | <b>71.8±2.6</b> | <b>80.9±1.7</b> | <b>85.3±3.2</b> |

#### AUC-ranked best-vs-second-best p-value analysis for Table 5.

P-values compare the AUC-best component setting with the AUC-second-best component setting using two-sided paired t-tests on the five-fold metrics.

| Comparison | Acc | F1 | Rec | Pre | AUC |
|---|---:|---:|---:|---:|---:|
| SGA+CBF Loss vs SGA-only | <0.0001 (***) | <0.0001 (***) | 0.0017 (**) | <0.0001 (***) | 0.0003 (***) |

Significance symbols: * p < 0.05, ** p < 0.01, *** p < 0.001, ns not significant.


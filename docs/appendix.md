---
layout: page
title: Appendix
---
## Model Hyperparameters

For each model, we defaulted to using weighted cross entropy as our loss function and Adam as our optimizer, with a learning rate of 0.0001 and
weight decay of 0.9. We also had a scheduler to lower the learning rate when loss plateaued for 15 epochs. Each model was run with 8 layers (except for
SAN, which only had 4 due to memory constraints) and a batch size of 32 (Princeton was 8 due to memory constraints). 

We ran our models on a mix of hardware, either a Nvidia A100 or a Nvidia RTX 3090.

## Tabular Results

We used 5 different models, and 3 different optimizations on each of them (mixing and matching).

These tables display the test results of our models. *Italics* indicate the best train result (not listed here), 
and **bold** indicates the best test result for that dataset.

### PascalVOC-SP (Macro F1 Score)

| Modifier Type        | GCN    | GatedGCN | GIN    | GAT    | GAT (DW) | SAN    |
|----------------------|--------|----------|--------|--------|---------------|--------|
| Baseline             | 0.0691 | 0.0885   | 0.1092 | 0.1462 | 0.0493        | 0.2006 |
| Edge Adding          | 0.0510 | 0.0808   | 0.1058 | 0.1468 | 0.0493        | 0.1923 |
| Laplacian PE         | 0.0787 | 0.0865   | 0.1112 | 0.1448 | 0.0493        | **0.2195** |
| Random Walk SE       | 0.0799 | 0.0814   | 0.1124 | 0.1761 | 0.0493        | 0.2157 |
| Laplacian PE + Edges | 0.0764 | 0.0843   | 0.1143 | 0.1376 | 0.0493        | *0.2156* |
| RWSE + Edges         | 0.0758 | 0.0850   | 0.1115 | 0.1792 | 0.0493        | 0.2105 |

### Peptides-functional (Average Precision)

| Modifier Type        | GCN    | GatedGCN | GIN    | GAT    | GAT (DW) | SAN    |
|----------------------|--------|----------|--------|--------|---------------|--------|
| Baseline             | 0.4976 | 0.5166   | 0.5455 | 0.5245 | 0.4758        | 0.5745 |
| Edge Adding          | 0.5214 | 0.5275   | 0.5540 | 0.5161 | 0.4609        | 0.5696 |
| Laplacian PE         | 0.5326 | 0.5397   | 0.5643 | 0.5386 | 0.4784        | **0.6126** |
| Random Walk SE       | 0.5211 | 0.5467   | 0.5689 | 0.5577 | 0.4860        | *0.5913* |
| Laplacian PE + Edges | 0.5133 | 0.5291   | 0.5634 | 0.5455 | 0.4957        | 0.6066 |
| RWSE + Edges         | 0.5241 | 0.4701   | 0.5641 | 0.5395 | 0.4943        | 0.6092 |

### Princeton Shape Benchmark (Average Precision)

| Modifier Type        | GCN    | GatedGCN | GIN    | GAT    | GAT (DW) | SAN    |
|----------------------|--------|----------|--------|--------|---------------|--------|
| Baseline             | 0.2228 | 0.1974   | 0.3440 | 0.2688 | 0.1649        | 0.3875 |
| Edge Adding          | 0.1941 | 0.1429   | 0.2599 | 0.2123 | 0.1526        | 0.4270 |
| Laplacian PE         | 0.3497 | 0.3843   | 0.3842 | 0.3472 | 0.2693        | *0.4268* |
| Random Walk SE       | 0.3195 | 0.3921   | 0.3020 | 0.2951 | 0.2761        | 0.4415 |
| Laplacian PE + Edges | 0.3834 | 0.3089   | 0.3768 | 0.3382 | 0.3079        | 0.4493 |
| RWSE + Edges         | 0.3327 | 0.3358   | 0.3932 | 0.3357 | 0.3003        | **0.4629** |
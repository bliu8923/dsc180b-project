## GNN Performance on Long Range Node Classification and Graph Classification

This repository holds the code to test 4 different neural network architectures
on 2 different long range datasets. 

Network architectures can be found under src/models, and test data (Cora) can
be found under the test directory.

To test the model's performance on a small dataset, use the docker repo b6liu/dsc180b (cpu or gpu for tag) and run:
```azure
python run.py --test True --bz (number)
```
IF ON DSMLP: Run a smaller bz if GAN errors, defaults to 32 for test. Also, we recommend using
16+ GB of ram as the networks tend to have large numbers of parameters (especially for GAN/SAN).

We would also recommend a GPU for running these tests/benchmarks, in this case, you should pull
the GPU docker image that has Cuda 11.7 support. (b6liu/dsc180b:gpu)

Different parameters can be run on the file as well.

```--datatype```: Where to extract dataset (LRGB for pascal and peptides, 3D for PSB)

```--dataset```: Dataset to run, currently only support all LRGB datasets. Defaults to PascalVOC-SP

```--model```: Model to run, currently GNN, GatedGCN, GIN, GAT, SAN

```--bz```: Batch size, defaults to 32

```--epoch```: Number of epochs to run the model

```--criterion```: Loss function, defaults to cross entropy 

```--optimizer```: Optimizer to use, defaults to adam

```--lr```: Learning rate, defaults to 0.0005

```--momentum```: Momentum term, defaults to 0.9

```--weight-decay```: Weight decay term, defaults to 5e-6

```--task```: task for network, defaults to node level

```--metric```: Accuracy metric to perform, defaults to macro f1, support for AP

```--gamma```: (SAN only) sparcity of attention, 0 indicates sparse attention while 1 indicates no bias

```--hidden```: Hidden parameters, made after linearly encoding data

```--scheduler```: Enable or disable scheduling on plateau

Shortcut methods have been added:

```--add_edges```: ratio of edges to be created (fake, random connections between nodes)

```--encode```: Positional encoding, we support "lap" for laplacian encoding or "walk" for RWSE

```--encode_k```: number of features to be added by encoding

And for partial (GAT supported, distance weighting):

```--partial```: Number of distance weighted layers, should be 1

```--space```: Spacial representation of data (2 for 2d, 3 for 3d, etc)

```--k```: for KNN in distance weighting

These are the recommended commands to run all datasets on the best models:

```python run.py --model san --dataset PascalVOC-SP --metric macrof1  --add_edges 1 --encode lap --encode_k 10```

```python run.py --model san --dataset peptides-func --task graph --metric ap  --encode lap --encode_k 10```

```python run.py --model san --datatype 3d --dataset psb --metric ap --encode walk --encode_k 10 --add_edges 1```

You can find the results in the results folder, under the model, timestamped.

### Citations
Thank you to Long Range Graph Benchmarks for the SAN implementation and datasets.
```
@article{dwivedi2022LRGB,
  title={Long Range Graph Benchmark}, 
  author={Dwivedi, Vijay Prakash and Rampášek, Ladislav and Galkin, Mikhail and Parviz, Ali and Wolf, Guy and Luu, Anh Tuan and Beaini, Dominique},
  journal={arXiv:2206.08164},
  year={2022}
}
```
And to GraphGPS, for many loss functions, SAN, and encoders:
```
@article{rampasek2022GPS,
  title={{Recipe for a General, Powerful, Scalable Graph Transformer}}, 
  author={Ladislav Ramp\'{a}\v{s}ek and Mikhail Galkin and Vijay Prakash Dwivedi and Anh Tuan Luu and Guy Wolf and Dominique Beaini},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```
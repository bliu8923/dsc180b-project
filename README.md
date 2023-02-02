## GNN Performance on Long Range Node Classification and Graph Classification

This repository holds the code to test 4 different neural network architectures
on 2 different long range datasets. 

Network architectures can be found under src/models, and test data (Cora) can
be found under the test directory.

To test the model's performance on a small dataset, run:
```azure
python run.py --test True
```

Different parameters can be run on the file as well.

```--dataset```: Dataset to run, currently only support all LRGB datasets. Defaults to PascalVOC-SP

```--model```: Model to run, support for all in files except GTN, defaults to SAN

```--hidden```: Number of hidden channels in the model

```--bz```: Batch size, defaults to 32

```--epoch```: Number of epochs to run the model

```--criterion```: Loss function, defaults to cross entropy

```--optimizer```: Optimizer to use, defaults to adam

```--lr```: Learning rate, defaults to 0.01

```--momentum```: Momentum term, defaults to 0.9

```--weight-decay```: Weight decay term, defaults to 5e-6

```--accuracy-metric```: Accuracy metric to perform, defaults to macro f1\


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
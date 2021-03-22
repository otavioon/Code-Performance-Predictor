# GNN Based Code Performance Predictor 

This repository requires Python 3.5+ and `unzip` package.

## Downloading required data

We use a GNN to try to learn to predict the performance difference of an application compiled with two set of optimizations using the LLVM compiler. The dataset used is described bellow. 

For that, we use the network described in the following scheme:

![alt text](https://github.com/vandersonmr/Code-Performance-Predictor/blob/master/network.png)

The implementation of the network is in GNN.py and it uses the Spektral framework (https://github.com/danielegrattarola/spektral)

To install requirements, simple use:

```shell
pip install -r requirements-gpu.txt
```

In data used can be cloned from [CCPE-DADOS repository](https://github.com/andrefz/ccpe-dados) and must be placed inside `./data/` directory. You can use the `fetch_and_process_ccpe_data.sh` script, located in `./data/` directory to download and unpack the data properly. The command would be:

```shell
./data/fetch_and_process_ccpe_data.sh
```

After, unzip the runtimes file, which contains information about the execution time of each application with each optimization sequences. This can be done using the following command:

```shell
unzip -d ./data/runtime ./data/runtime.zip
```

### Description of CCPE-DATA

[CCPE-DATA](https://github.com/andrefz/ccpe-dados) contains information about the computational graphs of ~300 applications, each one compiled with 100 different optimization sequences. The root of this information is, by default, located at `./data/ccpe-dados/`. The information is expressed in two representations:
* **CFG Representation**: Located at `./data/ccpe-dados/cfg.llvm/`, each YAML file correspond to an application, optimized with an optimization sequence, with the respective representation extracted with an LLVM pass. Each file contains a graph and a matrix that describes the features of each node in the graph. Each feature is a 67-element vector. 

* **CDFG Representation**: Located at `./data/ccpe-dados/cfg.programl/`, each YAML file correspond to an application, optimized with an optimization sequence, with the respective representation extracted with [programl](https://github.com/ChrisCummins/ProGraML/). Each file contains a graph, the edge's features and a matrix that describes the features of each node in the graph. Each feature is a 200-element vector, in the [inst2vec](https://github.com/spcl/ncc) representation. Each edge in graph may be a control, a data or a call edge, expressed by the edge's features.


## Datasets generation

Once data is downloaded, the representations must be processed and a dataset must be created to feed the GNN as input. The notebooks `generate_cfgs.ipynb` and `generate_cdfgs.ipynb` are used to create the datasets with CFG and CDFG representations, respectively.

......

## Training Graph Neural Networks (GNN) to predict performance

See notebook `train_GNN.ipynb`.

....

## License

This code is under MIT license and the dataset in Creative Commons Attribution 4.0 International.

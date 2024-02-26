# GINN: Graph-Informed Neural Networks

by [Stefano Berrone](https://www.researchgate.net/profile/Stefano-Berrone), [Francesco Della Santa](https://www.researchgate.net/profile/Francesco-Della-Santa), [Antonio Mastropietro](https://www.researchgate.net/profile/Antonio-Mastropietro), [Sandra Pieraccini](https://www.researchgate.net/profile/Sandra-Pieraccini), [Francesco Vaccarino](https://www.researchgate.net/profile/Francesco-Vaccarino).

![Example of the action of a GINN filter](https://www.mdpi.com/mathematics/mathematics-10-00786/article_deploy/html/images/mathematics-10-00786-g001-550.jpg)

In this repository, we publish the codes used to implement the Graph-Informed Neural Networks (GINNs) presented in the paper:
_Graph-Informed Neural Networks for Regressions on Graph-Structured Data_, Mathematics 2022, 10(5), 786; https://doi.org/10.3390/math10050786

The paper is **opens access** and you can find it [HERE](https://www.mdpi.com/2227-7390/10/5/786/htm).

This work introduces a new spatial-based graph convolutional layer, called Graph-Informed (GI) layer. The new architecture is specifically designed for regression tasks on graph-structured data that are not suitable for the well-known graph neural networks, such as the regression of functions with the domain and codomain defined on two sets of values for the vertices of a graph. In particular, a GI layer exploits the adjacent matrix of a given graph to define the unit connections in the neural network architecture, describing a new convolution operation for inputs associated with the vertices of the graph. 
In the paper, the GINN models show very good regression abilities and interesting potentialities on two maximum-flow test problems of stochastic flow networks and on a real-world application concerning the flux regression problem in underground networks of fractures.

**SPARSE AND DENSE OPERATIONS:** unfortunately the tensorflow version used at the moment of the code development (2.7.0) does not allow all the sparse tensor operations we need (in particular, a sparse version of the tensordot operation). Then, the code of the GI layer class works with dense tensors. As soon as the tools for sparse tensors will be available in tensorflow, we will update the code. _**Latest News (February 2024):** Sparse implementation is ready and it will be uploaded soon!_

![Example of GINN](https://www.mdpi.com/mathematics/mathematics-10-00786/article_deploy/html/images/mathematics-10-00786-g005-550.jpg)

## Table of Contents
- [License](https://github.com/Fra0013To/GINN/blob/main/README.md#license)
- [Requirements](https://github.com/Fra0013To/GINN/blob/main/README.md#requirements)
- [Getting Started](https://github.com/Fra0013To/GINN/blob/main/README.md#getting-started)
  - [Inputs/Outputs Description](https://github.com/Fra0013To/GINN/blob/main/README.md#inputsoutputs-description)
  - [Layer Initialization](https://github.com/Fra0013To/GINN/blob/main/README.md#layer-initialization)
  - [Run the Example](https://github.com/Fra0013To/GINN/edit/blob/README.md#run-the-example)
- [Citation](https://github.com/Fra0013To/GINN/edit/blob/README.md#citation)

## License
_GINN_ is released under the MIT License (refer to the [LICENSE file](https://github.com/Fra0013To/GINN/blob/main/LICENSE) for details).

## Requirements
- Numpy 1.22.1
- Scipy 1.7.3
- TensorFlow 2.7.0

**N.B.:** in the requirements we use tensorflow for CPUs but the codes work also with tensorflow for GPUs. The [requirements.txt file](https://github.com/Fra0013To/GINN/blob/main/requirements.txt) contains the required python modules (list above) and the corresponding dependencies.

## Getting Started
The GI layer can be used and added to a Keras model as any other Keras layer. In the following, we describe the inputs and outputs of a GI layer and we list the arguments for a GI layer initialization. Similar information is contained in the class code as comments/helps.

Then, we illustrate how to run an example of GINN construction, training and prediction.

### Inputs/Outputs Description
Given the adjacency matrix _A_ of shape (_N_, _N_) of a graph and the number of filters _F_ (i.e., the desired number of output features) the GI layer w.r.t. _A_ and _F_ returns an array of shape (?, _N_, _F_), for each batch of inputs of shape (?, _N_, _K_), where:
- _K_ is the number of input features per graph node;
- The symbol "?" denotes the batch size.

If _F = 1_ or a _pooling option_ is selected, the output tensor has shape (?, _N_). At the moment, a general pooling operation that returns _F'_ output features, _1 < F' < F_, is not implemented yet.

If a list of _m_ _highlighted nodes_ is given, the array returned by the layer has shape (?, _m_, _F_) or (?, _m_), where the output features are the ones related to the _m_ selected nodes of the graph (i.e., performs the mask operation illustrated in the paper).

### Layer Initialization
The _GraphInformed_ class, in [nnlayers module](https://github.com/Fra0013To/GINN/blob/main/nnlayers.py) of this repository, is defined as a subclass of [_tensorflow.keras.layers.Dense_](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense). Then, we list and describe only the new input arguments for the initialization. All the other arguments (e.g., _activation_, _kernel_initializer_, etc.) are inherited by the _Dense_ class.

- **adj_mat**: a dictionary describing the adjacency matrix. The dictionary must have the following keys and values:
  - _'keys'_: list of tuples (row,column) corresponding to nonzero elements of the adjacency matrix;
  - _'values'_: list of the numerical values associated to the tuples in _'keys'_;
  - _'shape'_: shape of the adjacency matrix.
  
  If you have the adjacency matrix saved as a _scipy.sparse_ matrix, you can use the _spars2dict_ function in the [utils module](https://github.com/Fra0013To/GINN/blob/main/utils.py) of this repository to convert it into a proper dictionary for the GraphInformed class. On the other hand, the function _dict2sparse_ returns a _scipy.sparse_ matrix from this kind of dictionaries;
- **num_filters**: the integer number _F_ of output features per node of the layer. Default is 1;
- **pool**: _None_ or a _string_ denoting a tf-reducing function (e.g.: ['reduce_mean'](https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean), ['reduce_max'](https://www.tensorflow.org/api_docs/python/tf/math/reduce_max), etc.). Default is _None_;
- **highlighted_nodes**: _None_ or list of _m <= N_ indexes of the graph nodes on which the layer is focused. Default is _None_.

### Run the Example
To see a code example of GINN construction, training and prediction, see the script [ginn_example.py](https://github.com/Fra0013To/GINN/blob/main/ginn_example.py) in this repository.  
To run the example (bash terminal):
1. Clone the repository:
    ```bash 
    git clone https://github.com/Fra0013To/GINN.git
    ```
2. Install the [required python modules](https://github.com/Fra0013To/GINN/edit/main/README.md#requirements).
    ```bash
    pip install -r requirements.txt
    ```
    or 
    ```bash
    pip install numpy==1.22.1
    pip install scipy==1.7.3
    pip install tensorflow==2.7.0
    ```
3. Run the script [ginn_example.py](https://github.com/Fra0013To/GINN/blob/main/ginn_example.py):
    ```bash
    python ginn_example.py
    ```

## Citation
If you find GINNs useful in your research, please cite:
#### BibTeX
> @Article{math10050786,  
> AUTHOR = {Berrone, Stefano and {Della Santa}, Francesco and Mastropietro, Antonio and Pieraccini, Sandra and Vaccarino, Francesco},  
> TITLE = {Graph-Informed Neural Networks for Regressions on Graph-Structured Data},  
> JOURNAL = {Mathematics},  
> VOLUME = {10},  
> YEAR = {2022},  
> NUMBER = {5},  
> ARTICLE-NUMBER = {786},  
> ISSN = {2227-7390},  
> DOI = {10.3390/math10050786}   
> }
#### RIS
> TY  - EJOU  
> AU  - Berrone, Stefano  
> AU  - Della Santa, Francesco  
> AU  - Mastropietro, Antonio  
> AU  - Pieraccini, Sandra  
> AU  - Vaccarino, Francesco  
> TI  - Graph-Informed Neural Networks for Regressions on Graph-Structured Data  
> T2  - Mathematics  
> PY  - 2022  
> VL  - 10  
> IS  - 5  
> SN  - 2227-7390  
> KW  - graph neural networks  
> KW  - deep learning  
> KW  - regression on graphs  
> DO  - 10.3390/math10050786  

## Update
- 2022.02.28: Repository creation.

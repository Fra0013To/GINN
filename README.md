# GINN: Graph-Instructed Neural Networks

by [Stefano Berrone](https://www.researchgate.net/profile/Stefano-Berrone), [Francesco Della Santa](https://www.researchgate.net/profile/Francesco-Della-Santa), [Antonio Mastropietro](https://www.researchgate.net/profile/Antonio-Mastropietro), [Sandra Pieraccini](https://www.researchgate.net/profile/Sandra-Pieraccini), [Francesco Vaccarino](https://www.researchgate.net/profile/Francesco-Vaccarino).

![Example of the action of a GI layer](https://www.mdpi.com/mathematics/mathematics-10-00786/article_deploy/html/images/mathematics-10-00786-g001-550.jpg)

Fig. 1: Graph-Instructed (GI) layer action

![Example of the action of a EWGI layer](./EWGIlayer.png)

Fig. 2: Edge-Wise Graph-Instructed (EWGI) layer action

In this repository, we publish the codes necessary to implement the Graph-Instructed Neural Networks (GINNs, formerly 
called _Graph-Informed Neural Networks_), presented 
for the first time in the paper: _Graph-Informed Neural Networks for Regressions on Graph-Structured Data_, Mathematics 
2022, 10(5), 786; https://doi.org/10.3390/math10050786

The papers related to GINNs are listed in the following
1. _Graph-Informed Neural Networks for Regressions on Graph-Structured Data_, Mathematics 2022, 
786; https://doi.org/10.3390/math10050786 **[OPEN ACCESS]**;
1. _Graph-Informed Neural Networks for Sparse Grid-Based Discontinuity Detectors_, arXiv preprint, https://arxiv.org/abs/2401.13652
1. _Sparse Implementation of Versatile Graph-Informed Layers_, arXiv preprint, http://arxiv.org/abs/2403.13781;
1. _Edge-Wise Graph-Instructed Neural Networks_, Journal of Computational Science 2025; https://doi.org/10.1016/j.jocs.2024.102518 **[OPEN ACCESS]**


The history of GINNs in brief:
- **2019**: Idea, Development and first results on Discrete Fracture Network problems;
- **Dec. 2019/Jan. 2020**: abstract subsmission of _Graph-Informed Neural Networks for Flux Regression in Discrete Fracture Networks_, for the [Computational Methods in Water Resources XXIII](https://cmwrconference.org/2020/) (CMWR),
Stanford (US);
- **Dec. 2020**: Presentation of _Graph-Informed Neural Networks for Flux Regression in Discrete 
Fracture Networks_, at the [Computational Methods in Water Resources XXIII](https://cmwrconference.org/2020/) (CMWR), 
remote, postponed w.r.t. June 2020 for COVID pandemic, [Young researcher presenter award 
winner](https://cmwrconference.org/2020/award-winners/);
- **Dec. 2021**: Presentation of _Predicting flux in Discrete Fracture Networks via Graph Informed Neural Networks_, 
at the [Machine Learning and the Physical Sciences, 2021](https://ml4physicalsciences.github.io/2021/), NeurIPS 
Foundation;
- **March 2022**: First publication on journal, _Graph-Informed Neural Networks for Regressions on Graph-Structured Data_, Mathematics 2022, 10(5), 
786; https://doi.org/10.3390/math10050786
- **2022 - Present**: See the list of papers above.
  - **Sept. 2024**: Change of name into _Graph-Instructed Neural Networks_.

Graph-Instructed (GI) layers are defined through a new spatial-based graph convolutional operation. The new architecture 
is specifically designed for regression tasks on graph-structured data that are not suitable for the well-known graph 
neural networks, such as the regression of functions with the domain and codomain defined on two sets of values for the 
vertices of a graph. In particular, a GI layer exploits the adjacent matrix of a given graph to define the unit 
connections in the neural network architecture, describing a new convolution operation for inputs associated with the 
vertices of the graph. 
In the original paper, the GINN models show very good regression abilities and interesting potentialities on two 
maximum-flow test problems of stochastic flow networks and on a real-world application concerning the flux regression 
problem in underground networks of fractures. In more recent works, GINNs have been applied also to classification tasks 
(e.g., detection of points near to discontinuity interfaces of functions, see https://arxiv.org/abs/2401.13652).

**SPARSE AND DENSE OPERATIONS:** sparse implementation of GI layers has been introduced after 
http://arxiv.org/abs/2403.13781 (March 2024) and is currently available! The dense implementation of the original GI 
layers' paper is still present in the repository, but it is deprecated.

**NOVEL EDGE-WISE GI LAYERS:** Edge-wise layer implementation is now available, introduced after _Edge-Wise 
Graph-Instructed Neural Networks_ (Journal of Computational Science, 2025). The implementation of EWGI layer is based on 
the new, sparse, GI layer implementation (v2.0 of this repository). 

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
- Numpy 1.25.2
- Pandas 2.2.3
- PyYAML 6.0.2
- Scikit-Learn 1.5.2
- Scipy 1.12.0
- TensorFlow 2.15.0.post1

**N.B.:** The [requirements.txt file](https://github.com/Fra0013To/GINN/blob/main/requirements.txt) contains the required python modules (list above).

## Getting Started
The GI layer can be used in Keras model as any other Keras layer. 

In the following, we describe the inputs and outputs 
of a GI layer and we list the arguments for a GI layer initialization. Similar information is contained in the class 
code as comments/helps. Then, we illustrate how to run an example of GINN construction, training and prediction.

For a full explanation of the current version of GI layers, see http://arxiv.org/abs/2403.13781. 

### Inputs/Outputs Description
Given the adjacency matrix _A_ of shape (_N_, _N_) of a graph and the number of filters _F_ (i.e., the desired number 
of output features) the GI layer w.r.t. _A_ and _F_ returns an array of shape (?, _N_, _F_), for each batch of inputs of shape (?, _N_, _K_), where:
- _K_ is the number of input features per graph node;
- The symbol "?" denotes the batch size.

If _F = 1_ or a _pooling option_ is selected, the output tensor has shape (?, _N_). At the moment, a general pooling 
operation that returns _F'_ output features, _1 < F' < F_, is not implemented yet.

The layer can be defined also with respect to a submatrix _A'_ of shape (_n1_, _n2_) of the adjacency matrix, 
representing the subgraph identified by two sets _V1_ and _V2_ of _n1<=N_, and _n2<=N_ graph nodes, respectively. 
In this case, the layer is characterized by the connections between the chosen subsets of nodes and the indices of these
nodes must be given during the initialization of the layer.

### Layer Initialization
The _GraphInformed_ class, in [graphinstructed.layers module](https://github.com/Fra0013To/GINN/blob/main/graphinstructed/layers.py) 
of this repository, is defined as a subclass of [_tensorflow.keras.layers.Dense_](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) (more precisily, it is a subclass
of the deprecated non-sparse implementation of original GI layers, that is a subclass of Dense layers). Then, we list 
and describe only the new input arguments for the initialization. All the other arguments 
(e.g., _activation_, _kernel_initializer_, etc.) are inherited by the _Dense_ class.

#### Initialization Arguments:
the arguments are listed in the same order of the code in the repository.
- **adj_mat**: the matrix _A'_. It must be a _scipy.sparse.dok_matrix_ or _scipy.sparse.dok_array_, or a dictionary 
describing the adjacency matrix using the following keys:
  - _keys_: list of tuples _(i,j)_ denoting the non-zero elements of the matrix _A'_;
  - _values_: list of non-zero values of _A'_, corresponding to _keys_; 
  - _rowkeys_custom_: list of indices _i(1),... ,_i(n1)_ denoting the nodes in _V1_. If _None_, we assume that they are 
  0,... , (n1 - 1);
  - _colkeys_custom_: list of indices _j(1),... ,_j(n2)_ denoting the nodes in _V2_. If _None_, we assume that they are 
  0,... , (n2 - 1);
  - _keys_custom_: list of tuples _(i(k),j(h))_ that "translate" the tuples in _keys_ with respect to the indices stored
  in _rowkeys_custom_ and _colkeys_custom_. If _None_, we assume that this list is equal to the one stored in _keys_. 

  Such a kind of dictionary can be easily obtained from a sparse matrix using the _sparse2dict_ function defined in 
the [graphinstructed.utils module](https://github.com/Fra0013To/GINN/blob/main/graphinstructed/utils.py) .
- **rowkeys**: list, default _None_. List containing the indices of the nodes in _V1_. If _None_, we assume that the 
indices are 0,... , (n1 - 1). Any list is automatically sorted in ascending order. This argument is ignored if the 
_adj_mat_ argument is a dictionary.
- **colkeys**: list, default _None_. List containing the indices of the nodes in _V2_. If _None_, we assume that the 
indices are 0,... , (n2 - 1). Any list is automatically sorted in ascending order. This argument is ignored if the 
_adj_mat_ argument is a dictionary.
- **selfloop_value**: float, default 1.0. Rescaling factor of the self-loop connections added by the graph-convolution 
operation. Modify this value only if _A_ is the adjacency matrix of a weighted graph and you want specific weights for 
self-loops added by the graph-convolution.
- **num_filters**: integer, default 1. Integer value describing the number _F_ of filters (i.e., output features per 
node) of the layer. The value _K_ of input features per node is inferred directly from the inputs.
- **activation**, **use_bias**, **kernel_initializer**, **bias_initializer**, **kernel_regularizer**, 
**bias_regularizer**, **activity_regularizer**, **kernel_constraint**, **bias_constraint**: see the _tensorflow.keras.layers.Dense_ class;
- **pool**: string, default _None_. String describing a "reducing function" of tensorflow (e.g., 'reduce_mean', 
'reduce_max', etc.);

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
    pip install numpy==1.25.2
    pip pandas==2.2.3
    pip PyYAML==6.0.2
    pip scikit_learn==1.5.2
    pip install scipy==1.12.0
    pip install tensorflow==2.15.0.post1
    ```
3. Run the script [ginn_example.py](https://github.com/Fra0013To/GINN/blob/main/ginn_example.py):
    ```bash
    python ginn_example.py
    ```

## Citation
If you find GINNs useful in your research, please cite the following papers (BibTeX and RIS versions):
#### BibTeX
> @Article{math10050786,  
> TITLE = {Graph-Informed Neural Networks for Regressions on Graph-Structured Data},  
> AUTHOR = {Berrone, Stefano and {Della Santa}, Francesco and Mastropietro, Antonio and Pieraccini, Sandra and Vaccarino, Francesco},
> JOURNAL = {Mathematics},  
> VOLUME = {10},  
> YEAR = {2022},  
> NUMBER = {5},  
> ARTICLE-NUMBER = {786},  
> ISSN = {2227-7390},  
> DOI = {10.3390/math10050786}   
> }
   
> @misc{dellasanta2024sparse,  
>       title={Sparse Implementation of Versatile Graph-Informed Layers},   
>       author={{Della Santa}, Francesco},  
>       year={2024},  
>       eprint={2403.13781},  
>       archivePrefix={arXiv},  
>       primaryClass={cs.LG},  
>       url={https://arxiv.org/abs/2403.13781 },  
>       doi={10.48550/arXiv.2403.13781}  
> }

> @article{DELLASANTA2025ewginn,  
> title = {Edge-Wise Graph-Instructed Neural Networks},  
> journal = {Journal of Computational Science},  
> volume = {85},  
> pages = {102518},  
> year = {2025},  
> issn = {1877-7503},  
> doi = {10.1016/j.jocs.2024.102518},  
> url = {https://www.sciencedirect.com/science/article/pii/S1877750324003119 },
> author = {Francesco {Della Santa} and Antonio Mastropietro and Sandra Pieraccini and Francesco Vaccarino},  
> keywords = {Graph neural networks, Deep learning, Regression on graphs},  
> }


#### RIS
> TY  - EJOU  
> TI  - Graph-Informed Neural Networks for Regressions on Graph-Structured Data  
> JO  - Mathematics  
> AU  - Berrone, Stefano  
> AU  - Della Santa, Francesco  
> AU  - Mastropietro, Antonio  
> AU  - Pieraccini, Sandra  
> AU  - Vaccarino, Francesco  
> PY  - 2022  
> VL  - 10  
> IS  - 5  
> SN  - 2227-7390  
> KW  - graph neural networks  
> KW  - deep learning  
> KW  - regression on graphs  
> DO  - 10.3390/math10050786 

> TY  - EJOU  
> TI  - Sparse Implementation of Versatile Graph-Informed Layers   
> T2  - arXiv  
> AU  - Della Santa, Francesco  
> PY  - 2024   
> KW  - graph neural networks  
> KW  - deep learning  
> DO  - https://doi.org/10.48550/arXiv.2403.13781

> TY  - JOUR  
> T1  - Edge-Wise Graph-Instructed Neural Networks  
> AU  - Della Santa, Francesco  
> AU  - Mastropietro, Antonio  
> AU  - Pieraccini, Sandra  
> AU  - Vaccarino, Francesco  
> JO  - Journal of Computational Science  
> VL  - 85  
> SP  - 102518  
> PY  - 2025  
> DA  - 2025/02/01/  
> SN  - 1877-7503  
> DO  - https://doi.org/10.1016/j.jocs.2024.102518  
> UR  - https://www.sciencedirect.com/science/article/pii/S1877750324003119  
> KW  - Graph neural networks  
> KW  - Deep learning  
> KW  - Regression on graphs  
> ER  -   

## Updates and Versions
- v 3.1 (2024.12.02): Edge-Wise GI layer implementation (Sparse, based on GI layer of v 2.0, see https://doi.org/10.1016/j.jocs.2024.102518  )
- v 3.0 (2024.09.12): Edge-Wise GI layer implementation (Dense, based on GI layer of v 1.0)
- v 2.0 (2024.03.22): Sparse implementation and versatile general form of GI layers (see http://arxiv.org/abs/2403.13781).
- v 1.0 (2022.02.28): Repository creation.

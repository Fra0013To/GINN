# GINN: Graph-Informed Neural Networks

by Stefano Berrone, Francesco Della Santa, Antonio Mastropietro, Sandra Pieraccini, Francesco Vaccarino.

In this repository, we publish the codes used to implement the Graph-Informed Neural Networks (GINNs) presented in the paper:
_Graph-Informed Neural Networks for Regressions on Graph-Structured Data_, Mathematics 2022, 10(5), 786; https://doi.org/10.3390/math10050786 (registering DOI)

The paper is **opens access** and you can find it [HERE](https://www.mdpi.com/2227-7390/10/5/786/htm).

This work introduces a new spatial-based graph convolutional layer, called Graph-Informed (GI) layer. The new architecture is specifically designed for regression tasks on graph-structured data that are not suitable for the well-known graph neural networks, such as the regression of functions with the domain and codomain defined on two sets of values for the vertices of a graph. In particular, a GI layer exploits the adjacent matrix of a given graph to define the unit connections in the neural network architecture, describing a new convolution operation for inputs associated with the vertices of the graph. 
In the paper, the GINN models show very good regression abilities and interesting potentialities on two maximum-flow test problems of stochastic flow networks and on a real-world application concerning the flux regression problem in underground networks of fractures.


## License
_GINN_ is released under the MIT License (refer to the LICENSE file for details).

## Citation
If you find GINNs useful in your research, please cite:
#### BibTeX
> @Article{math10050786,  
> AUTHOR = {Berrone, Stefano and Della Santa, Francesco and Mastropietro, Antonio and Pieraccini, Sandra and Vaccarino, Francesco},  
> TITLE = {Graph-Informed Neural Networks for Regressions on Graph-Structured Data},  
> JOURNAL = {Mathematics},  
> VOLUME = {10},  
> YEAR = {2022},  
> NUMBER = {5},  
> ARTICLE-NUMBER = {786},  
> URL = {https://www.mdpi.com/2227-7390/10/5/786},  
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

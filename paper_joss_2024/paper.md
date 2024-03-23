---
title: 'Sparse Implementation of Versatile Graph-Informed Layers'
tags:
  - Python
  - Tensorflow
  - Deep Learning
  - Graph Neural Networks
authors:
  - name: Francesco Della Santa
    orcid: 0000-0002-2202-9600
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Mathematical Sciences, Politecnico di Torino, Turin, Italy
   index: 1
 - name: Member of the INdAM-GNCS research group, Rome, Italy
   index: 2
date: DD Mont YYYY
bibliography: paper.bib

---

# Abstract

Graph Neural Networks (GNNs) have emerged as effective tools for learning tasks on graph-structured data. Recently, Graph-Informed (GI) layers were introduced to address regression tasks on graph nodes, extending their applicability beyond classic GNNs. However, existing implementations of GI layers lack efficiency due to dense memory allocation. This paper presents a sparse implementation of GI layers, leveraging the sparsity of adjacency matrices to reduce memory usage significantly. Additionally, a versatile general form of GI layers is introduced, enabling their application to subsets of graph nodes. The proposed sparse implementation improves the concrete computational efficiency and scalability of the GI layers, permitting to build deeper Graph-Informed Neural Networks (GINNs) and facilitating their scalability to larger graphs.

# Introduction

Graph Neural Networks (GNNs) are well known as powerful tools for learning tasks on graph-structured data [@GNNsurvey2020], such as semi-supervised node classification, link prediction, and graph classification, with their origin that dates back to the late 2000s [@firstGNN_Gori2005],[@firstGNN_Micheli2009],[@firstGNN_Scarselli2009]. Recently, a new type of layer for GNNs called Graph-Informed (GI) layer [@GINN] has been developed, specifically designed for regression tasks on graph-nodes; indeed, this type of task is not suitable for classic GNNs and, therefore, typically it is approached using MLPs, that do not exploit the graph structure of the data. Nonetheless, the usage of GI layers has been recently extended also to supervised classification tasks (see [@dellasanta2024graphinformed]).

The main advantages of the GI layers is the possibility to build Neural Networks (NNs), called Graph-Informed NNs (GINNs), suitable for large graphs and deep architectures. Their good performances, especially if compared with respect to classic MLPs, are illustrated both in [@GINN] (regression tasks) and [@dellasanta2024graphinformed] (classification task for discontinuity detection).

However, at the time this work is written, existing GI layer implementations have one main limitation. Specifically, all the variables in the codes do not exploit the sparsity of the adjacency matrix of the graph. Therefore, the memory allocated to store a GI layer is much larger, because all the zeros (representing missing connections between graph nodes) are stored even if they are not involved in the GI layer's graph-convolution operation. The problem of this ``dense'' implementation is that the computer's memory can be easily saturated, especially when building GINNs based on large graphs and/or many GI layers (i.e., deep architectures). Therefore, the principal scope of this work is to present a new implementation of the GI layers that is sparse; i.e., an implementation that exploits the sparsity of the adjacency graph, reducing concretely the memory allocated for storing GI layers and GINNs in general.

In [@GINN], the definition of a GI layer is very general and can be easily applied to any kind of graph that is directed or not and that is weighted or not. However, the original definition can be further generalized, extending the action of GI layers to subsets of graph nodes, introducing few more details. In this work, we introduce such a kind of generalization, called _versatile general form_ of GI layers.

Summarizing, the proposed implementation of GI layers that is illustrated in this work allows the handling of sub-graphs and is optimized to leverage the sparse nature of adjacency matrices, resulting in significant improvements in computational efficiency and memory utilization. In particular, the sparse implementation enables the construction of deeper GINNs and facilitates scalability for larger graphs.

The work is organized as follows: in [#graph-informed-layers] the GI layers are formally introduced, recalling their inner mechanisms. In [#Versatile GI Layers] the versatile general form of GI layers is formally defined, explaining similarities and differences with respect to the previous version of GI layers. In [#Sparse Versatile GI Layers in Tensorflow] we describe in details the sparse implementation of the versatile GI layers, both reporting the pseudocode ([#Pseudocode]) and the documentation of the python class available in the public repository ([https://github.com/Fra0013To/GINN], march 2024 update). In (#Conclusion) we summarize the results of the work.


# Graph-Informed Layers

# Versatile GI Layers

# Sparse Versatile GI Layers in Tensorflow

## Pseudocode

## Usage Documentation

# Conclusion

# Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References

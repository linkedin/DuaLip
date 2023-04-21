<p align="center">
   <img src="https://github.com/linkedin/DuaLip/blob/master/logo.jpg" height="300">
</p>
   
# DuaLip: Dual Decomposition based Linear Program Solver
[![Build Status](https://github.com/linkedin/DuaLip/actions/workflows/ci.yml/badge.svg?branch=master&event=push)](https://github.com/linkedin/DuaLip/actions/workflows/ci.yml?query=branch%3Amaster+event%3Apush)
[![Release](https://img.shields.io/github/v/release/linkedin/DuaLip)](https://github.com/linkedin/DuaLip/releases/)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](LICENSE)

DuaLip is an **extreme-scale** Linear Program (LP) solver based on Apache Spark. It solves structured LP 
problems of the following form arising from web-applications:

```
minimize        c'x
subject to      Ax <= b
                x_i in C_i  for i in 1,2,...,I
```

where `x = (x_1, ..., x_I)` is the full vector of optimization variables, `x_i` is the vector of optimization
variables associated with one `i`, and `A`,`b`,`c` and `C_i` are 
user-supplied data.  
 
It is a distributed solver that solves a perturbation of the LP problem at scale 
via gradient-based algorithms on the smooth dual of the perturbed LP with 
computational guarantees. DuaLip can easily scale to problems in trillions of variables. 

This library was created by [Yao Pan](https://www.linkedin.com/in/panyaopy/), [Kinjal Basu](https://www.linkedin.com/in/kinjalbasu/), 
[Rohan Ramanath](https://www.linkedin.com/in/rohanramanath/), [Konstantin Salomatin](https://www.linkedin.com/in/ksalomatin/), [Amol Ghoting](https://www.linkedin.com/in/amolghoting/), [Sathiya Keerthi](https://www.linkedin.com/in/sathiya-keerthi-selvaraj-ba963414/) and 
[Miao Cheng](https://www.linkedin.com/in/miaoch/) from LinkedIn.

## Copyright

Copyright 2022 LinkedIn Corporation
All Rights Reserved.

Licensed under the BSD 2-Clause License (the "License").
See [License](LICENSE) in the project root for license information.

## Features

### Extreme Scale
DuaLip is specifically developed to tackle problems arising in web applications that usually have hundreds of millions of users
and millions of items, pushing the number of optimization variables in the trillions range (if not more). It uses a dual 
decomposition technique to be able to scale to such large problems. For details and a wide range of applications, see [Ramanath et. al. (2021)](https://arxiv.org/pdf/2103.05277.pdf) and [Basu et. al. (2020)](http://proceedings.mlr.press/v119/basu20a/basu20a.pdf).

### Parallelism
We support extreme-scale parallelism in our DuaLip solver, which can solve tens of millions of small separate LPs simultaneously. Such problems arise in applications like personalized constrained optimization in recommender systems which have personalized constraints for each unique user. In this case, each LP is dedicated to each user and only contains a small number of decision variables. Our parallel version of DuaLip can solve tens of millions of separate small LPs for all different users in parallel.

### Efficient
Although we use well-known first-order gradient methods to solve the problem, we implement several highly efficient algorithms 
for each of the component steps. This allows us to scale up 20x over a naive 
implementation. Please see [Ramanath et. al. (2021)](https://arxiv.org/pdf/2103.05277.pdf) for a comparative study.

### Modular Design
In our implementation, any problem can be formulated through a highly modular approach.
* `solver`: We begin by choosing a first-order optimization solver. We currently support
[Accelerated Gradient Ascent](https://www.ceremade.dauphine.fr/~carlier/FISTA), [LBFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) and [LBFGS-B](https://en.wikipedia.org/wiki/Limited-memory_BFGS).
* `projectionType`: We implement several very efficient projection algorithms to allow for a wide class of constraint
sets `C_i`. 

Each of these components is highly flexible and can be easily customized to add new solvers, or new types of projections
for different constraints sets `C_i`. New formulations can also be added by appropriately stitching together these different components.

### Detects Infeasibility
We have incorporated simple checks on infeasibility. This helps the end user to appropriately tweak their problem space.    

### Extensive Logging
We have added extensive logging to help users understand whether the solver has converged to a good approximate solution. 

### Warm start
We allow the user to input an initial estimate of the dual solution if they have one (e.g., if they have solved a highly related problem before). This can result in very efficient solving of the overall problem.
 
For more details of these features please see the full [wiki](https://linkedin.github.io/DuaLip/). 

## Usage

### Building the Library
It is recommended to use Scala 2.12 and Spark 3.1.1. To build, run the following:
```bash
./gradlew build
```
This will produce a JAR file in the ``./dualip/build/libs/`` directory.

Tests typically run with the `test` task. If you want to force-run all tests, you can use:
```bash
./gradlew cleanTest test --no-build-cache
```

### Using the JAR File
Depending on the mode of usage, the built JAR can be deployed as part of an offline data pipeline, depended 
upon to build jobs using its APIs, or added to the classpath of a Spark Jupyter notebook or a Spark Shell instance. For
example:
```bash
$SPARK_HOME/bin/spark-shell --jars target/dualip_2.12.jar
```

### Usage Examples
For detailed example usage, please see the [Getting Started](https://linkedin.github.io/DuaLip/get_started/index.html) wiki.

## Contributions
If you would like to contribute to this project, please review the instructions [here](contributions.md).

## Acknowledgments
Implementations of some methods in DuaLip were inspired by other open-source libraries. Discussions with several LinkedIn employees influenced
aspects of this library. A full list of acknowledgements can be found [here](acknowledgements.md).

## References
DuaLip has been created on the basis of the following research papers. If you cite DuaLip, please use the following:
```
@inproceedings{ramanath:21,
  author       = {Ramanath, Rohan, and Keerthi, Sathiya S. and Basu, Kinjal and Salomatin, Konstantin and Yao, Pan},
  title        = {Efficient Algorithms for Global Inference in Internet Marketplaces},
  journal      = {arXiv preprint arXiv:2103.05277},
  year         = {2021},
  url          = {https://arxiv.org/abs/2103.05277}
}

@InProceedings{pmlr-v119-basu20a,
  title        = {{ECLIPSE}: An Extreme-Scale Linear Program Solver for Web-Applications},
  author       = {Basu, Kinjal and Ghoting, Amol and Mazumder, Rahul and Pan, Yao},
  booktitle    = {Proceedings of the 37th International Conference on Machine Learning},
  pages        = {704--714},
  year         = {2020},
  volume       = {119},
  series       = {Proceedings of Machine Learning Research},
  month        = {13--18 Jul},
  publisher    = {PMLR},
  pdf          = {http://proceedings.mlr.press/v119/basu20a/basu20a.pdf},
  url          = {http://proceedings.mlr.press/v119/basu20a.html}
}

@misc{dualip,
    author     = {Ramanath, Rohan, and Keerthi, Sathiya S. and Basu, Kinjal and Salomatin, Konstantin and Yao, Pan and Ghoting, Amol and Cheng, Miao},
    title      = {{DuaLip}: Dual Decomposition based Linear Program Solver, version 2.0.0},
    url        = {https://github.com/linkedin/dualip},
    month      = dec,
    year       = 2022
}
```


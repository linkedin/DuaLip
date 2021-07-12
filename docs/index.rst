Dualip: Dual Decomposition based Linear Program Solver Documentation
====================================================================

DuaLiP is an extreme-scale Linear Program (LP) solver based on Apache Spark. It solves structured LP problems of the following form arising from web-applications:

.. math::
  \begin{array}{ll}
    \mbox{minimize} & c^T x \\
    \mbox{subject to} & Ax \leq b \\
    &x_i \in \mathcal{C}_i \;\;\text{for all} ~i\in \{1,2,...,I\}
  \end{array}

where :math:`x = (x_1, ..., x_I)` is the full vector of optimization variables, :math:`x_i` is the vector of optimization
variables associated with one :math:`i`, and :math:`A,b,c` and :math:`C_i` are user-supplied data.

It is a distributed solver that solves a perturbation of the LP problem at scale via gradient-based algorithms on the smooth dual of the perturbed LP with computational guarantees. Dualip can easily scale to problems in trillions of variables.

This library was created by `Yao Pan
<https://www.linkedin.com/in/panyaopy/>`_, `Kinjal Basu
<https://www.linkedin.com/in/kinjalbasu/>`_, `Rohan Ramanath
<https://www.linkedin.com/in/rohanramanath/>`_, `Konstantin Salomatin
<https://www.linkedin.com/in/ksalomatin/>`_, `Amol Ghoting
<https://www.linkedin.com/in/amolghoting/>`_, and `Sathiya Keerthi
<https://www.linkedin.com/in/sathiya-keerthi-selvaraj-ba963414/>`_ from LinkedIn.

**Code available on** `GitHub 
<https://github.com/linkedin/dualip>`_.

Citing Dualip
------------------

If you are using Dualip for your work, we encourage you to

* :ref:`Cite the related papers <citing>`
* Put a star on GitHub |github-star|


.. |github-star| image:: https://img.shields.io/github/stars/oxfordcontrol/osqp.svg?style=social&label=Star
  :target: https://github.com/linkedin/dualip


**We are looking forward to hearing your success stories with Dualip!** Please `share them with us
<dualip@linkedin.com>`_.

Features
------------------

.. glossary::

#. Extreme Scale
    Dualip is specifically developed to tackle problems arising in web-applications that usually have hundreds of millions of users and millions of items, pushing the number of optimization variables in the trillions range (if not more). It uses a dual decomposition technique to be able to scale to such large problems. For details see the :ref:`solution <probsolution>` and for a wide range of applications see `Ramanath et. al. (2021) <https://arxiv.org/abs/2103.05277>`_ and `Basu et. al (2020) <http://proceedings.mlr.press/v119/basu20a/basu20a.pdf>`_.

#. Efficient
    Although we follow first-order gradient methods to solve the problem, we implement several highly efficient algorithms for each of the component steps. See the :ref:`solver <solver>` for more details. This allows us to scale up 20x over a naive implementation. Please see `Ramanath et. al. (2021) <https://arxiv.org/abs/2103.05277>`_ for a comparative study.

#. Modular Design
    In our implementation, any problem can be formulated through a highly modular approach.

		- **solver**: We begin by choosing a first-order optimization solver. We currently support `Proximal Gradient Ascent <https://en.wikipedia.org/wiki/Proximal_gradient_method>`_, `Accelerated Gradient Ascent <https://www.ceremade.dauphine.fr/~carlier/FISTA>`_, and `LBFGS-B <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`_
		- **projectionType**: We implement several very efficient projection algorithms to allow for a wide class of constraint sets :math:`\mathcal{C}_i`. For a list of supported constraints sets see :ref:`here <constraints>`.

    Each of these components is highly flexible and can be easily customized to add new solvers, or new types of projections for different constraints sets :math:`\mathcal{C}_i`. New formulations can also be added by appropriately stitching together these different components.

#. Detects Infeasibility
    We have incorporated simple checks on infeasibility (see Appendix D of `our paper <https://arxiv.org/abs/2103.05277>`_). This helps the end user to appropriately tweak the problem space.

#. Extensive Logging
    We have added extensive logging to help users understand whether the solver has converged to a good approximate solution. For more details, please see :ref:`here <logging>`.

#. Warm start
    We allow the user to input an initial estimate of the dual solution, if she is familiar with the problem space. This allows for very efficient solving of the overall problem.

#. Autotuning
    We have implemented an autotuning mechanism, that automatically selects and adaptively changes the smoothness parameter to achieve faster convergence. In most situations, users may not know how to choose a value for this parameter and we alleviate this issue completely.

Usage
------------------
Building the Library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It is recommended to use Scala 2.11.8 and Spark 2.3.0. To build, run the following:

.. code:: bash

  ./gradlew build

This will produce a JAR file in the :code:`./dualip/build/libs/` directory.
If you want to use the library with Spark 2.4 (and the Scala 2.11.8 default), you can specify this when running the build command.

.. code:: bash

  ./gradlew build -PsparkVersion=2.4.3

Tests typically run with the :code:`test` task. If you want to force-run all tests, you can use:

.. code:: bash

  ./gradlew cleanTest test --no-build-cache

Using the JAR File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Depending on the mode of usage, the built JAR can be deployed as part of an offline data pipeline, depended 
upon to build jobs using its APIs, or added to the classpath of a Spark Jupyter notebook or a Spark Shell instance. For example:

.. code:: bash

  $SPARK_HOME/bin/spark-shell --jars target/dualip_2.11.jar


Copyright
------------------
Copyright 2021 LinkedIn Corporation All Rights Reserved.
Licensed under the BSD 2-Clause License (the "License"). See License in the project root for license information.

Contributions
------------------
If you would like to contribute to this project, please review the instructions :ref:`here <contributions>`.

Bug reports and support
-----------------------
Please report any issues via the `Github issue tracker <https://github.com/linkedin/dualip/issues>`_. All types of issues are welcome including bug reports, documentation typos, feature requests and so on.

Acknowledgments
------------------
Implementations of some methods in DuaLip were inspired by other open-source libraries. Discussions with several LinkedIn employees influenced aspects of this library. A full list of acknowledgements can be found :ref:`here <acknowledgements>`.

References
------------------
DuaLip has been created on the basis of the following research paper. If you cite DuaLip, please use the following:

        .. code:: latex

          @inproceedings{ramanath:21,
            author       = {Ramanath, Rohan, and Keerthi, Sathiya S. and Basu, Kinjal and Salomatin, Konstantin and Yao, Pan},
            title        = {Efficient Algorithms for Global Inference in Internet Marketplaces},
            journal      = {arXiv preprint arXiv:2103.05277},
            year         = {2021},
            url          = {https://arxiv.org/abs/2103.05277}
            }


          @InProceedings{pmlr-v119-basu20a,
            title      = {{ECLIPSE}: An Extreme-Scale Linear Program Solver for Web-Applications},
            author     = {Basu, Kinjal and Ghoting, Amol and Mazumder, Rahul and Pan, Yao},
            booktitle  = {Proceedings of the 37th International Conference on Machine Learning},
            pages      = {704--714},
            year       = {2020},
            volume     = {119},
            series     = {Proceedings of Machine Learning Research},
            month      = {13--18 Jul},
            publisher  = {PMLR},
            pdf        = {http://proceedings.mlr.press/v119/basu20a/basu20a.pdf},
            url        = {http://proceedings.mlr.press/v119/basu20a.html}
          }

          @misc{dualip,
            author       = {Ramanath, Rohan, and Keerthi, Sathiya S. and Basu, Kinjal and Salomatin, Konstantin and Yao, Pan and Ghoting, Amol},
            title        = {{DuaLip}: Dual Decomposition based Linear Program Solver, version 1.0.0},
            url          = {https://github.com/linkedin/dualip},
            month        = mar,
            year         = 2021
          }


        

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Documentation

   solver/index
   get_started/index
   interfaces/index
   demo/index
   citing/index
   contributing/index
   acknowledgement/index


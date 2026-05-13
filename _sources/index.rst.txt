DuaLip: Dual Decomposition-based LP Solver
======================================================

DuaLip is an extreme-scale Linear Program (LP) solver based on Pytorch. It solves structured LP problems of the following form arising from web-applications:

.. math::
  \begin{array}{ll}
    \mbox{minimize} & c^T x \\
    \mbox{subject to} & Ax \leq b \\
    &x_i \in \mathcal{C}_i \;\;\text{for all} ~i\in \{1,2,...,I\}
  \end{array}

where :math:`x = (x_1, ..., x_I)` is the full vector of optimization variables, :math:`x_i` is the vector of optimization
variables associated with one :math:`i` (e.g., user :math:`i`), and :math:`A,b,c` and :math:`C_i` are problem-specific data.

DuaLip solves a ridge-regularized (perturbed) LP by applying gradient-based methods to the resulting smooth dual, with provable computational guarantees. It can easily scale to problems in trillions of variables thanks to its parallelized implementation and multi-GPU support. DuaLip offers several features to make it suitable for extreme-scale LP problems:

.. rst-class:: glossary-features
   :sorted:

  Extreme Scale
    DuaLip is specifically developed to tackle problems arising in web applications with hundreds of millions of users and millions of items, where the induced optimization problems can involve trillions of variables (or more). This scale is achieved through a combination of inherently parallelizable dual decomposition algorithms and careful exploitation of modern GPU-based computing systems.
    See the :ref:`Problem Solution <probsolution>` section for details. For a wide range of applications, see `Ramanath et. al. (2021) <https://arxiv.org/abs/2103.05277>`_ and `Basu et. al (2020) <http://proceedings.mlr.press/v119/basu20a/basu20a.pdf>`_.

  Operator-centric programming model
    Problem logic is expressed only through three primitives:

    - **ObjectiveFunction**: which encapsulates the data and dual gradient computation,
    - **ProjectionMap**: which performs blockwise projections onto constraint polytopes :math:`C_i`,
    - **Maximizer**: which performs dual ascent using the dual gradient.

  The modular design allows for easy extension to new constraint families and formulations as well as switching between different optimization algorithms for the Maximizer without modifying the core solver logic.
  
  Efficient
    We enhance the underlying solver by incorporating Jacobi-style preconditioning of the constraint matrix and adaptive smoothing that progressively decays the regularization parameter to improve convergence rate and robustness in large-scale, real-world workloads.  
  
  Extensive Logging
    We provide extensive MLFlow logs to help users understand whether the solver has converged to a good approximate solution. For more details, please see :ref:`here <logging>`.

  Warm start
    We allow the user to input an initial estimate of the dual solution if they have one (e.g., if they have solved a highly related problem before). This can result in very efficient solving of the overall problem.


**Code available on** `GitHub
<https://github.com/linkedin/DuaLip>`_.

.. container:: custom-title
   
   Copyright

Copyright 2022 LinkedIn Corporation All Rights Reserved.
Licensed under the BSD 2-Clause License (the "License"). See License in the project root for license information.

.. container:: custom-title
   
   Contributing

If you would like to contribute to this project, please review the instructions :ref:`here <contribution>`.

.. container:: custom-title
   
   Bug reports and support

Please report any issues via the `Github issue tracker <https://github.com/linkedin/dualip/issues>`_. All types of issues are welcome including bug reports, documentation typos, feature requests and so on.

.. container:: custom-title
   
   Citing DuaLip

If you are using DuaLip for your work, we encourage you to

* :ref:`Cite the related papers <citing>`
* Put a star on GitHub |github-star|

.. |github-star| image:: https://img.shields.io/github/stars/linkedin?style=social
  :target: https://github.com/linkedin/DuaLip


**We are looking forward to hearing your success stories with DuaLip!** Please `share them with us
<ask_lp@linkedin.com>`_.

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
            author       = {Ramanath, Rohan, and Keerthi, Sathiya S. and Basu, Kinjal and Salomatin, Konstantin and Yao, Pan and Ghoting, Amol and Cheng, Miao},
            title        = {{DuaLip}: Dual Decomposition based Linear Program Solver, version 2.0.0},
            url          = {https://github.com/linkedin/dualip},
            month        = dec,
            year         = 2022
          }




.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Documentation

   solver/index
   interfaces/index
   get_started/index
   demo/index
   citing/index
   contributing/index
   acknowledgement/index


Getting Started
===============

Building the Library
--------------------

It is recommended to use Scala 2.12 and Spark 3.1.1. To build, run the following:

.. code:: bash

  ./gradlew build

This will produce a JAR file in the :code:`./dualip/build/libs/` directory.

Tests typically run with the :code:`test` task. If you want to force-run all tests, you can use:

.. code:: bash

  ./gradlew cleanTest test --no-build-cache


Add a DuaLip Dependency to Your Project
---------------------------------------

Please check `Artifactory
<https://linkedin.jfrog.io/artifactory/DuaLip/>`_ for the latest artifact versions.

Gradle Example
^^^^^^^^^^^^^^

The artifacts are available in LinkedIn's Artifactory instance and in Maven Central, so you can specify either repository in the top-level build.gradle file.

.. code:: java

  repositories {
      mavenCentral()
      maven {
          url "https://linkedin.jfrog.io/artifactory/open-source/"
      }
  }

Add the DuaLip dependency to the module-level :code:`build.gradle` file. Here are some examples for multiple recent
Spark/Scala version combinations:

.. code:: java

  dependencies {
      compile 'com.linkedin.dualip:dualip_3.1.1_2.12:2.4.8'
  }

  dependencies {
      compile 'com.linkedin.dualip:dualip_3.1.1_2.12:2.4.6'
  }
  
  dependencies {
      compile 'com.linkedin.dualip:dualip_3.1.1_2.12:2.4.2'
  }


Using the JAR File
------------------

Depending on the mode of usage, the built JAR can be deployed as part of an offline data pipeline, depended 
upon to build jobs using its APIs, or added to the classpath of a Spark Jupyter notebook or a Spark Shell instance. For
example:

.. code:: bash

  $SPARK_HOME/bin/spark-shell --jars target/dualip_2.12.jar


Usage Examples
--------------
Currently the library supports two different solvers:

1. :code:`MooSolver`: This solves multi-objective optimization problems, which include a
few global or cohort-level constraints and are characterized by small number of rows 
in :math:`A` (usually less than one hundred)

2. :code:`MatchingSolver`: This solves matching problems, where we have a large number of 
per-item constraints. The number of rows of :math:`A` here is quite large and can range up to
1 million.

Both the solvers support a wide range of constraints :math:`\mathcal{C}_i` as seen `here
<../solver/index.html#constraints>`_
as well as a wide variety of `first-order optimization methods
<../solver/index.html#algorithm>`_.

There is a unified driver implementation :code:`com.linkedin.dualip.solver.LPSolverDriver` for 
both of these problems which serves as the primary entry point.

We currently support a parallel version of the :code:`MooSolver`, which can solve many separate 
small Moo problems in parallel. The number of such small Moo problems can range up to tens of millions.

Call :code:`com.linkedin.dualip.solver.ParallelLPSolverDriver` to leverage the extreme-scale parallelism 
power of our DuaLip solver.

For detailed usage please see the :ref:`Parameters` and the :ref:`demo`.
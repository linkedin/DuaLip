

Parameters
=============

Solver Parameters 
-----------------
The solver settings are displayed in the following table.

================================ =============  ==============  ====================================================================================
Parameters                       Is Required    Default Value   Description
================================ =============  ==============  ====================================================================================
:code:`driver.objectiveClass`    true           N/A             Objective function implementation
:code:`driver.solverOutputPath`  true           N/A             The outputPath of solver
:code:`driver.initialLambdaPath` false          N/A             Optional path to initialize dual variables for algorithm restarts
:code:`driver.gamma`             false          1E-3            Coefficient for quadratic objective regularizer, used by most objectives
:code:`driver.outputFormat`      false          AVRO            The format of output, can be AVRO, JSON or ORC
:code:`driver.projectionType`    true           N/A             One of available projections (simplex, unitbox, etc.)
:code:`driver.savePrimal`        false          false           Flag to save primal
:code:`driver.verbosity`         false          1               The levels of logging to be shown. We currently support 0 and 1.
:code:`input.ACblocksPath`       true           N/A             Path of matrix A & c encoded as data blocks
:code:`input.vectorBPath`        true           N/A             Path of vector of budgets b
:code:`input.metadataPath`       true           N/A             Path for the metadata of the problem
:code:`input.format`             true           N/A             The format of input data, e.g. AVRO, JSON or ORC
:code:`optimizer.maxIter`        true           N/A 			The maximum number of iterations the solver will run
:code:`optimizer.solverType`     true           N/A             The type of optimizer, currently supported: AGD, LBFGSB
:code:`optimizer.dualTolerance`  true           N/A             Tolerance criteria for dual variable change
:code:`optimizer.slackTolerance` true           N/A             Tolerance criteria for slack
================================ =============  ==============  ====================================================================================


Spark Parameters 
----------------
There are four regular spark parameters you can tune: 

* :code:`spark.driver.mem`
* :code:`spark.executor.mem`
* :code:`spark.executor.cores`
* :code:`spark.executor.num`

The most efficient spark parameters are dependent on the problem scale and structure and there is no one-fit-all 
configurations. The default parameters usually can achieve acceptable performance, but users can definitely try tuning them if needed.  

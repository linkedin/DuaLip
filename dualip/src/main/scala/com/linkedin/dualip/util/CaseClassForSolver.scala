package com.linkedin.dualip.util

import com.linkedin.dualip.util.DataFormat.DataFormat
import com.linkedin.dualip.util.Status.Status

/**
  * Define the metaData for solver input
  *
  * @param numRows     - Number of rows in Matrix A
  * @param numCols     - Number of columns in Matrix A
  */
case class MetaData(numRows: Long, numCols: Long)

/**
  * Case class to represent input path parameters
  *
  * @param ACblocksPath    - Path of matrix A & c encoded as data blocks
  * @param vectorBPath     - Path of vector of budgets b (this should be a dense vector, every itemId should have a constraint)
  * @param format          - The format of input data, e.g. avro or orc
  */
case class InputPaths(ACblocksPath: String, vectorBPath: String, format: DataFormat)

/**
  * Case class to represent the solution of the solver
  *
  * @param x - Primal solution
  * @param y - dual solution
  */
case class Solution(x: String, y: String)

/**
  * Case class to represent optimizer state
  *
  * @param iterations    - Number of iterations taken
  * @param status        - optimizer status, could be Running, Converged, Infeasible, Terminated or Failed
  * @param log           - log from optimizer
  */
case class OptimizerState(iterations: Int, status: Status, log: String)

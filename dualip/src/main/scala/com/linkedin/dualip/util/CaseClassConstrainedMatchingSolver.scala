package com.linkedin.dualip.util

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.util.DataFormat.DataFormat

/**
  * case class representing costs
  *
  * @param costLocal  : contribution to the objective by a local constraint
  * @param costGlobal : contribution to the objective by a set of global constraints
  */
case class ConstrainedMatchingCosts(costLocal: (Int, Double), costGlobal: Seq[(Int, Double)])

/**
  * slate for constrained matching problem
  *
  * @param x         : primal variable
  * @param objective : value of objective associated with primal variable
  * @param costs     : costs associated with primal variable
  */
case class ConstrainedMatchingSlate(x: Double, objective: Double, costs: ConstrainedMatchingCosts)

/**
  * case class for the dual variables corresponding to local and global constraints
  *
  * @param lambdaLocal  : dual variables corresponding to local constraints
  * @param lambdaGlobal : dual variables corresponding to global constraints
  */
case class ConstrainedMatchingDuals(lambdaLocal: Array[Double], lambdaGlobal: Array[Double])

/**
  * case class for the dual variables corresponding to local and global constraints
  *
  * @param lambdaLocal  : dual variables corresponding to local constraints in BSV format
  * @param lambdaGlobal : dual variables corresponding to global constraints in BSV format
  */
case class ConstrainedMatchingDualsBSV(lambdaLocal: BSV[Double], lambdaGlobal: BSV[Double])

/**
  * case class for constrained matching solver parameters
  *
  * @param constrainedMatchingDataPath : Path of A matrix, G matrix and c vector combined in a special data block
  * @param localBudgetPath             : Path of the budgets corresponding to local constraints
  * @param globalBudgetPath            : Path of the budgets corresponding to global constraints
  * @param format                      : The format of input data, e.g. avro or orc
  * @param numOfPartitions             : number of partitions for sp
  * @param enableHighDimOptimization   : enables high-dimensional optimization
  * @param numLambdaPartitions         : number of partitions for the duals
  */
case class ConstrainedMatchingSolverParams(
  constrainedMatchingDataPath: String,
  localBudgetPath: String,
  globalBudgetPath: String,
  format: DataFormat,
  numOfPartitions: Int,
  enableHighDimOptimization: Boolean = false,
  numLambdaPartitions: Option[Int] = None)

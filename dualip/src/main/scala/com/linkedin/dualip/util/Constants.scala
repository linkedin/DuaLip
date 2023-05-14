package com.linkedin.dualip.util

import com.linkedin.dualip.util

/**
  * The enumeration of supported file format.
  */
object DataFormat extends Enumeration {
  type DataFormat = Value
  val AVRO: util.DataFormat.Value = Value("avro")
  val ORC: util.DataFormat.Value = Value("orc")
  val JSON: util.DataFormat.Value = Value("json")
  val CSV: util.DataFormat.Value = Value("csv")
}

/**
  * The enumeration of available optimizers
  */
object OptimizerType extends Enumeration {
  type OptimizerType = Value
  val LBFGSB: util.OptimizerType.Value = Value("LBFGSB")
  val LBFGS: util.OptimizerType.Value = Value("LBFGS")
  val AGD: util.OptimizerType.Value = Value("AGD")
  val GD: util.OptimizerType.Value = Value("GD")
  val SUBGD: util.OptimizerType.Value = Value("SUBGD")
}

/**
  * The enumeration of available projections
  */
object ProjectionType extends Enumeration {
  type ProjectionType = Value
  val Greedy: util.ProjectionType.Value = Value("greedy") // Pick the item with the largest reward for each data block
  val SecondPrice: util.ProjectionType.Value = Value("secondPrice") // Used for greedy allocation in a multi-slot case
  val Simplex: util.ProjectionType.Value = Value("simplex") // As defined in SimplexProjection, \sum_j x_j = 1
  val SimplexInequality: util.ProjectionType.Value = Value("simplexInequality") // As defined in SimplexProjection, \sum_j x_j <= 1
  val BoxCut: util.ProjectionType.Value = Value("boxCut") // As defined in BoxSimplexProjection, \sum_j x_j = k
  val BoxCutInequality: util.ProjectionType.Value = Value("boxCutInequality") // As defined in BoxSimplexProjection, \sum_j x_j <= k
  val UnitBox: util.ProjectionType.Value = Value("unitBox") // As defined in UnitBoxProjection, 0 <= x_j <= 1
}

/**
  * The enumeration of return status.
  */
object Status extends Enumeration {
  type Status = Value
  val Running: util.Status.Value = Value("Running")
  val Converged: util.Status.Value = Value("Converged")
  val Infeasible: util.Status.Value = Value("Infeasible")
  val Terminated: util.Status.Value = Value("Terminated")
  val Failed: util.Status.Value = Value("Failed")
}
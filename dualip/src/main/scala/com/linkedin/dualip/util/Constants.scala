package com.linkedin.dualip.util

/**
  * The enumeration of supported file format.
  */
object DataFormat extends Enumeration {
  type DataFormat = Value
  val AVRO = Value("avro")
  val ORC = Value("orc")
  val JSON = Value("json")
  val CSV = Value("csv")
}

/**
  * The enumeration of available optimizers
  */
object OptimizerType extends Enumeration {
  type OptimizerType = Value
  val LBFGSB = Value("LBFGSB")
  val LBFGS = Value("LBFGS")
  val AGD = Value("AGD")
  val GD = Value("GD")
}

/**
  * The enumeration of available projections
  */
object ProjectionType extends Enumeration {
  type ProjectionType = Value
  val Greedy = Value("greedy")    // Pick the item with the largest reward for each data block
  val SecondPrice = Value("secondPrice") // Used for greedy allocation in a multi-slot case
  val Simplex = Value("simplex")  // As defined in SimplexProjection, \sum_j x_j = 1
  val SimplexInequality = Value("simplexInequality")  // As defined in SimplexProjection, \sum_j x_j <= 1
  val BoxCut = Value("boxCut")  // As defined in BoxSimplexProjection, \sum_j x_j = k
  val BoxCutInequality = Value("boxCutInequality")  // As defined in BoxSimplexProjection, \sum_j x_j <= k
  val UnitBox = Value("unitBox")  // As defined in UnitBoxProjection, 0 <= x_j <= 1
}

/**
  * The enumeration of return status.
  */
object Status extends Enumeration {
  type Status = Value
  val Running = Value("Running")
  val Converged = Value("Converged")
  val Infeasible = Value("Infeasible")
  val Terminated = Value("Terminated")
  val Failed = Value("Failed")
}
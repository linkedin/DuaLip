package com.linkedin.dualip.driver

import com.linkedin.dualip.util.DataFormat.DataFormat
import com.linkedin.dualip.util.ProjectionType.{ProjectionType, Simplex}
import com.linkedin.dualip.util.{DataFormat, ProjectionType}

/**
  * @param initialLambdaPath   Filepath to initialize dual variables (lambda) for algorithm restarts (optional).
  * @param initialLambdaFormat The format of file with initial lambda values, e.g. "avro" (default), "json", "orc"
  *                            or "csv".
  * @param autotune            Flag to algorithmically choose regularization parameter gamma (default is false).
  * @param gamma               Coefficient for quadratic objective regularizer, used by most objectives
  *                            (default is 1E-3).
  * @param projectionType      Type of projection used (default is Simplex).
  * @param boxCutUpperBound    Upper bound for the box cut projection constraint (default is 1).
  * @param objectiveClass      Class name of the objective function. The class should be a subclass of
  *                            DualPrimalObjectiveLoader. One of "MooSolverDualObjectiveFunction",
  *                            "MatchingSolverDualObjectiveFunction", "ConstrainedMatchingSolverDualObjectiveFunction",
  *                            "ParallelMooSolverDualObjectiveFunction", "MultiSlateMatchingSolverDualObjectiveFunction".
  * @param outputFormat        The format of output, can be "avro" (default) or "orc".
  * @param savePrimal          Flag to save primal variable values at the solution (default is false).
  * @param verbosity           0: Concise logging. 1: log with increased verbosity. 2: log everything.
  * @param solverOutputPath    Directory path to save solution at (default is "").
  */
case class LPSolverDriverParams(initialLambdaPath: Option[String] = None,
  initialLambdaFormat: DataFormat = DataFormat.AVRO,
  autotune: Boolean = false,
  gamma: Double = 1E-3,
  projectionType: ProjectionType = Simplex,
  boxCutUpperBound: Int = 1,
  objectiveClass: String = "",
  outputFormat: DataFormat = DataFormat.AVRO,
  savePrimal: Boolean = false,
  verbosity: Int = 1,
  solverOutputPath: String = ""
)

/**
  * Driver parameters parser
  */
object LPSolverDriverParamsParser {
  def parseArgs(args: Array[String]): LPSolverDriverParams = {
    val parser = new scopt.OptionParser[LPSolverDriverParams]("Parsing solver parameters") {
      override def errorOnUnknownArgument = false

      opt[String]("driver.initialLambdaPath") optional() action {
        (x, c) => c.copy(initialLambdaPath = Option(x))
      }
      opt[String]("driver.initialLambdaFormat") optional() action {
        (x, c) => c.copy(initialLambdaFormat = DataFormat.withName(x))
      }
      opt[Boolean]("driver.autotune") optional() action { (x, c) => c.copy(autotune = x) }
      opt[Double]("driver.gamma") optional() action { (x, c) => c.copy(gamma = x) }
      opt[String]("driver.projectionType") required() action {
        (x, c) => c.copy(projectionType = ProjectionType.withName(x))
      }
      opt[Int]("driver.boxCutUpperBound") optional() action { (x, c) => c.copy(boxCutUpperBound = x) }
      opt[String]("driver.objectiveClass") required() action { (x, c) => c.copy(objectiveClass = x) }
      opt[String]("driver.outputFormat") optional() action {
        (x, c) => c.copy(outputFormat = DataFormat.withName(x))
      }
      opt[Boolean](name = "driver.savePrimal") optional() action { (x, c) => c.copy(savePrimal = x) }
      opt[Int](name = "driver.verbosity") optional() action { (x, c) => c.copy(verbosity = x) }
      opt[String]("driver.solverOutputPath") required() action { (x, c) => c.copy(solverOutputPath = x) }
    }

    parser.parse(args, LPSolverDriverParams()) match {
      case Some(params) => params
      case _ => throw new IllegalArgumentException(s"Parsing the command line arguments ${args.mkString(", ")} failed")
    }
  }
}

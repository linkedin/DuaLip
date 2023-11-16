package com.linkedin.dualip.driver

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.data.MooData
import com.linkedin.dualip.driver.LPSolverDriver.{getInitialLambda, solve}
import com.linkedin.dualip.maximizer.{DualPrimalMaximizer, DualPrimalMaximizerLoader}
import com.linkedin.dualip.objective.DualPrimalObjective
import com.linkedin.dualip.problem.{MooSolverDualObjectiveFunction, ParallelMooSolverDualObjectiveFunction}
import com.linkedin.dualip.util.{InputPathParamsParser, MapReduceArray}
import com.linkedin.dualip.util.VectorOperations.toBSV
import org.apache.spark.sql.{Dataset, SaveMode, SparkSession}

/**
  * Entry point to Parallel LP solver. It leverages the LPSolverDriver for solving a single LP, which
  * initializes the necessary components:
  *   - objective function
  *   - optimizer
  *   - initial value of lambda
  *   - output parameters
  *     and runs the solver
  */
object ParallelLPSolverDriver {

  /**
    * Entry point to spark job
    *
    * @param args
    */
  def main(args: Array[String]): Unit = {

    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName(getClass.getSimpleName)
      .getOrCreate()

    import spark.implicits._

    try {
      val driverParams = LPSolverDriverParamsParser.parseArgs(args)
      val inputParams = InputPathParamsParser.parseArgs(args)
      println(driverParams)
      println(inputParams)

      val setOfProblems: Dataset[(Long, MapReduceArray[MooData], Array[(Int, Double)])] =
        ParallelMooSolverDualObjectiveFunction.loadData(inputParams, driverParams.gamma, driverParams.projectionType)

      val solver: DualPrimalMaximizer = DualPrimalMaximizerLoader.load(args)

      val resultsDS = setOfProblems.map { case (problemId, data, budget) =>
        val objective: DualPrimalObjective =
          new MooSolverDualObjectiveFunction(data, toBSV(budget, budget.length), driverParams.gamma, driverParams.projectionType, driverParams.boxCutUpperBound)
        // initialize lambda: first try custom logic of the objective, then driver-generic lambda loader, finally initialize with zeros
        val initialLambda: BSV[Double] = objective.getInitialLambda.getOrElse(
          getInitialLambda(driverParams.initialLambdaPath, driverParams.initialLambdaFormat, objective.dualDimensionality)
        )
        val results = solve(objective, solver, initialLambda, driverParams, parallelMode = true)

        (problemId, results.logList, results.dualList, results.violationList)
      }.toDF("problemId", "logList", "dualList", "violationList")

      resultsDS
        .repartition(1000)
        .write
        .format(driverParams.outputFormat.toString)
        .mode(SaveMode.Overwrite)
        .save(driverParams.solverOutputPath)

    } catch {
      case other: Exception => sys.error("Got an exception: " + other)
    } finally {
      spark.stop()
    }
  }
}

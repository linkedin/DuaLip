/*
 * BSD 2-CLAUSE LICENSE
 *
 * Copyright 2021 LinkedIn Corporation
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

package com.linkedin.dualip.solver

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.problem.MatchingSolverDualObjectiveFunction.toBSV
import com.linkedin.dualip.problem.{MooDataBlock, MooSolverDualObjectiveFunction, ParallelMooSolverDualObjectiveFunction}
import com.linkedin.dualip.solver.LPSolverDriver.{getInitialLambda, solve}
import com.linkedin.dualip.util.{InputPathParamsParser, MapReduceArray}
import org.apache.spark.sql.{Dataset, SaveMode, SparkSession}
import org.apache.log4j.Logger

/**
 * Entry point to Parallel LP solver. It leverages the LPSolverDriver for solving a single LP, which
 * initializes the necessary components:
 *   - objective function
 *   - optimizer
 *   - initial value of lambda
 *   - output parameters
 * and runs the solver
 */
object ParallelLPSolverDriver {

  val logger: Logger = Logger.getLogger(getClass)

  /**
   * Entry point to spark job
   * @param args
   */
  def main(args: Array[String]): Unit = {

    implicit val spark = SparkSession
      .builder()
      .appName(getClass.getSimpleName)
      .getOrCreate()

    import spark.implicits._

    try {
      val driverParams = LPSolverDriverParamsParser.parseArgs(args)
      val inputParams = InputPathParamsParser.parseArgs(args)
      println(driverParams)
      println(inputParams)

      val setOfProblems: Dataset[(Long, MapReduceArray[MooDataBlock], Array[(Int, Double)])] =
        ParallelMooSolverDualObjectiveFunction.loadData(inputParams, driverParams.gamma, driverParams.projectionType)

      val solver: DualPrimalGradientMaximizer = DualPrimalGradientMaximizerLoader.load(args)

      val resultsDS = setOfProblems.map { case (problemId, data, budget) =>
        val objective: DualPrimalDifferentiableObjective =
          new MooSolverDualObjectiveFunction(data, toBSV(budget, budget.length), driverParams.gamma, driverParams.projectionType, parallelMode = true)
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
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
 
package com.linkedin.dualip.problem

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.projection.{GreedyProjection, SimplexProjection, UnitBoxProjection}
import com.linkedin.dualip.slate.{DataBlock, SingleSlotOptimizer, Slate, SlateOptimizer}
import com.linkedin.dualip.solver.{DistributedRegularizedObjective, DualPrimalDifferentiableObjective, DualPrimalObjectiveLoader, 
  PartialPrimalStats}
import com.linkedin.dualip.util.{IOUtility, InputPaths}
import com.linkedin.dualip.util.ProjectionType._
import com.twitter.algebird.{Max, Tuple5Semigroup}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

import scala.util.Try

/**
  * Objective function for matching slate solver.
  *
  *
  * The class is declared serializable only for slatOptimizer serialization
  * @param problemDesign  - parallelized problem representation
  * @param b              - constraints vector
  * @param slateOptimizer - algorithm to generate primal given dual
  * @param gamma          - behaves like a regularizer and controls the smoothness of the objective
  * @param enableHighDimOptimization - passthrough parameter to the parent class (spark optimization for very high dimensional problems)
  * @param spark
  */
class MatchingSolverDualObjectiveFunction(
  problemDesign: Dataset[DataBlock],
  b: BSV[Double],
  slateOptimizer: SlateOptimizer,
  gamma: Double,
  enableHighDimOptimization: Boolean
)(implicit spark: SparkSession) extends DistributedRegularizedObjective(b, gamma, enableHighDimOptimization) with Serializable {
  import spark.implicits._

  lazy val upperBound = slateOptimizer match {
    case singleSlotOptimizer: SingleSlotOptimizer => singleSlotOptimizer.getProjection match {
      case _: SimplexProjection => problemDesign.map(_.data.map { case (j, c, a) => c }.max + gamma / 2).reduce(_ + _)
      case _ => Double.PositiveInfinity
    }
    case _ => Double.PositiveInfinity
  }

  override def getPrimalUpperBound: Double = upperBound

  override def getSardBound(lambda: BSV[Double]): Double = {
    val lambdaArray: Broadcast[Array[Double]] = spark.sparkContext.broadcast(lambda.toArray) // for performance
    val aggregator = new Tuple5Semigroup[Int, Int, Double, Max[Double], Max[Int]]
    val (nonVertexSoln, numI, corralSize, corralSizeMax, jMax) = problemDesign.map { case block =>
      // below line is the same cost as a projection
      val (nV, corral, jM) = slateOptimizer.sardBound(block, lambdaArray.value)
      (nV, 1, corral, Max(corral), jM)
    }.reduce(aggregator.plus(_, _))
    println(f"percent_vertex_soln: ${100.0 * (numI - nonVertexSoln) / numI}\t" +
      f"avg_corral_size: ${corralSize / numI}\t" +
      f"max_corral_size:${corralSizeMax.get}")
    0.5 * nonVertexSoln * (1 - 1.0 / jMax.get)
  }

   /**
   * Convert slates (primal solution) into sufficient statistics of the solution.
   * @param lambda
   * @return
   */
  override def getPrimalStats(lambda: BSV[Double]): Dataset[PartialPrimalStats] = {
    getPrimal(lambda).flatMap { case (id, slates) =>
      slates.map { slate =>
        PartialPrimalStats(slate.costs.toMap, slate.objective, slate.x * slate.x)
      }
    }
  }

  /**
    * Get the primal value for a given dual variable. For matching solver primal is a dataset of slates
    * @param lambda
    * @return dataset of slates
    */
  def getPrimal(lambda: BSV[Double]): Dataset[(String, Seq[Slate])] = {
    val lambdaArray: Array[Double] = lambda.toArray // for performance
    problemDesign.map { case block =>
      (block.id, slateOptimizer.optimize(block, lambdaArray))
    }
  }

  /**
   * Get primal for saving. The schema is simplified for readability of clients:
   * some fields are dropped and some renamed:
   * {
   *    blockId: String, // often corresponds to impression in matching problems
   *    variables: Array[
   *      {
   *         value: Double // the value of the variable in primal solution, can be fractional
   *                       // in matching problems we usually expect variables in a block to
   *                       // sum to 1.0. More than one non-zero variable can have probabilistic
   *                       // allocation interpretation.
   *         items: Array[Int] // item ids in the variable. Often a single element if
   *                           // we select one item per request. But may be a ranked list of items
   *                           // if we need to fill a multi-slot slate.
   *      }
   *    ]
   * }
   * Note. There is a potential optimization to use last primal computed during the optimization.
   * Unlikely to help a lot - cost is equivalent to one extra iteration.
   * todo: consider case class to define return DataFrame schema
   * @param lambda
   * @return Optionally the DataFrame with primal solution. None if the functionality is not supported.
   */
  override def getPrimalForSaving(lambda: BSV[Double]): Option[DataFrame] = {
    val renamedSchema = "array<struct<value:double,items:array<int>>>"
    val primal = getPrimal(lambda).map { case (blockId, slates) =>
      val variables = slates.map { s =>
        val items = s.costs.map { case (itemId, cost) => itemId }
        (s.x, items)
      }
      (blockId, variables)
    }.toDF("blockId", "variables")
      .withColumn("variables", col("variables").cast(renamedSchema))
    Option(primal)
  }
}

/**
 * Special parameters only for Matching optimizer
 * @param enableHighDimOptimization - spark optimization parameter for gradient computation
 *                                    set to true for very high dimensional lambdas (maybe >100K or 1M).
 *                                    and if each iteration is too slow or driver crashes.
 *                                    Default value is false
 */
case class MatchingSolverParams(slateSize: Int = 1, enableHighDimOptimization: Boolean = false)

/**
 * Companion object to load objective function from HDFS
 */
object MatchingSolverDualObjectiveFunction extends DualPrimalObjectiveLoader {
  /**
   * Load the problem objective and constraints
   * @param inputPaths
   * @param spark
   * @return
   */
  def loadData(
    inputPaths: InputPaths)(implicit spark: SparkSession): (Dataset[DataBlock], BSV[Double]) = {

    import spark.implicits._

    val budget = IOUtility.readDataFrame(inputPaths.vectorBPath, inputPaths.format)
      .map{case Row(_c0: Number, _c1: Number) => (_c0.intValue(), _c1.doubleValue()) }
      .collect

    val itemIds = budget.toMap.keySet
    // Check if every item has budget information encoded.
    (0 until budget.length).foreach { i: Int =>
      require(itemIds.contains(i), f"$i index does not have a specified constraint" )
    }

    val b = toBSV(budget, budget.length)

    var blocks = IOUtility.readDataFrame(inputPaths.ACblocksPath, inputPaths.format)
    // Make the optional fields of the DataBlock null in the dataframe
    DataBlock.optionalFields.foreach{
      field => if (Try(blocks(field)).isFailure) {
        blocks = blocks.withColumn(field, lit(null))
      }
    }
    val data = blocks.as[DataBlock]
      .repartition(spark.sparkContext.defaultParallelism)
      .persist(StorageLevel.MEMORY_ONLY)

    (data, b)
  }

  /**
   * Code to initialize slate optimizer. 
   * todo: Currently the available slate optimizers are hardcoded, consider an option to provide custom optimizer
   * @param gamma - gamma regularization (some optimizers require it)
   * @param slateSize - slate size
   * @param projectionType - one of available projections (simplex, unitbox, et.c.)
   * @return
   */
  def slateOptimizerChooser(gamma: Double, slateSize: Int, projectionType: ProjectionType): SlateOptimizer = {
    projectionType match {
      case Simplex => {
        require (slateSize == 1, "Single slot simplex algorithm requires matching.slateSize = 1")
        require (gamma > 0, "Gamma should be > 0 for simplex algorithm")
        new SingleSlotOptimizer(gamma, new SimplexProjection())
      }
      case SimplexInequality => {
        require(slateSize == 1, "Single slot inequality simplex algorithm requires matching.slateSize = 1")
        require(gamma > 0, "Gamma should be > 0 for simplex algorithm")
        new SingleSlotOptimizer(gamma, new SimplexProjection(inequality = true))
      }
      case UnitBox => {
        require (slateSize == 1, "Single slot unit box algorithm requires matching.slateSize = 1")
        require (gamma > 0, "Gamma should be > 0 for unit box projection algorithm")
        new SingleSlotOptimizer(gamma, new UnitBoxProjection())
      }
      case Greedy => {
        require (gamma == 0, "Gamma should be zero for max element slate optimizer")
        require (slateSize == 1, "Single slot algorithm requires matching.slateSize = 1")
        new SingleSlotOptimizer(gamma, new GreedyProjection())
      }
    }
  }

  /**
   * objective loader that conforms to a generic loader API
   * @param gamma
   * @param args
   * @param spark
   * @return
   */
  def apply(gamma: Double, projectionType: ProjectionType, args: Array[String])(implicit spark: SparkSession): DualPrimalDifferentiableObjective = {
    val inputPaths = InputPathParamsParser.parseArgs(args)
    val matchingParams = MatchingParamsParser.parseArgs(args)
    val (data, b) = loadData(inputPaths)
    val slateOptimizer: SlateOptimizer = slateOptimizerChooser(gamma, matchingParams.slateSize, projectionType)
    new MatchingSolverDualObjectiveFunction(data, b, slateOptimizer, gamma, matchingParams.enableHighDimOptimization)
  }

  /**
   * Utility method to convert array represented sparse vector to breeze sparse vector
   */
  def toBSV(data: Array[(Int, Double)], size: Int): BSV[Double] = {
    val (indices, values) = data.sortBy { case (index, _) => index }.unzip
    new BSV(indices, values, size)
  }
}

/**
 * Parameters parser
 */
object MatchingParamsParser {
  def parseArgs(args: Array[String]): MatchingSolverParams = {
    val parser = new scopt.OptionParser[MatchingSolverParams]("Matching slate solver params parser") {
      override def errorOnUnknownArgument = false
      opt[Int]("matching.slateSize") required() action { (x, c) => c.copy(slateSize = x) }
      opt[Boolean]("matching.enableHighDimOptimization") optional() action { (x, c) => c.copy(enableHighDimOptimization = x) }
    }

    parser.parse(args, MatchingSolverParams()) match {
      case Some(params) => params
      case _ => throw new IllegalArgumentException(s"Parsing the command line arguments ${args.mkString(", ")} failed")
    }
  }
}
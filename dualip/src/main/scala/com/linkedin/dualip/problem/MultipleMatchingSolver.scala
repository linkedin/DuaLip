package com.linkedin.dualip.problem

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.data.MultipleMatchingData
import com.linkedin.dualip.objective.distributedobjective.DistributedRegularizedObjective
import com.linkedin.dualip.objective.{DualPrimalObjective, DualPrimalObjectiveLoader, PartialPrimalStats}
import com.linkedin.dualip.projection.{BoxCutProjection, GreedyProjection, SimplexProjection, UnitBoxProjection}
import com.linkedin.dualip.slate.{MultipleMatchingSlateComposer, Slate}
import com.linkedin.dualip.util.ProjectionType._
import com.linkedin.dualip.util.{IOUtility, InputPathParamsParser, InputPaths}
import com.twitter.algebird.{Max, Tuple5Semigroup}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.storage.StorageLevel

import scala.util.Try

/**
  * Objective function for matching slate solver.
  *
  *
  * The class is declared serializable only for slateOptimizer serialization
  *
  * @param problemDesign                 - parallelized problem representation
  * @param budget                        - constraints vector
  * @param matchingConstraintsPerIndex   - number of matching constraints per index
  * @param multipleMatchingSlateComposer - algorithm to generate primal given dual
  * @param gamma                         - behaves like a regularizer and controls the smoothness of the objective
  * @param enableHighDimOptimization     - passthrough parameter to the parent class (spark optimization for very high dimensional problems)
  * @param numLambdaPartitions           - used when enableHighDimOptimization=true, dense lambda vectors coming from executors are partitioned
  *                                      for aggregation. The number of partitions should depend on aggregation parallelism and the dimensionality
  *                                      of lambda. A good rule of thumb is to use a multiple of aggregation parallelism to ensure even load
  *                                      but not too high to keep individual partition sizes large (e.g. 1000) for efficiency:
  *                                      numLambdaPartitions \in [5*parallelism, dim(lambda)/1000]
  * @param spark
  */
class MultipleMatchingSolverDualObjectiveFunction(problemDesign: Dataset[MultipleMatchingData],
  budget: BSV[Double],
  matchingConstraintsPerIndex: Int,
  multipleMatchingSlateComposer: MultipleMatchingSlateComposer,
  gamma: Double,
  enableHighDimOptimization: Boolean,
  numLambdaPartitions: Option[Int])
  (implicit spark: SparkSession) extends
  DistributedRegularizedObjective(budget, gamma, enableHighDimOptimization, numLambdaPartitions) with Serializable {

  import spark.implicits._

  lazy val upperBound: Double = Double.PositiveInfinity

  override def getPrimalUpperBound: Double = upperBound

  override def getSardBound(duals: BSV[Double]): Double = {
    val dualsArray: Broadcast[Array[Double]] = spark.sparkContext.broadcast(duals.toArray) // for performance
    val aggregator = new Tuple5Semigroup[Int, Int, Double, Max[Double], Max[Int]]

    val (nonVertexSoln, numI, corralSize, corralSizeMax, jMax) = problemDesign.map { block =>
      // below line is the same cost as a projection
      val (nV, corral, jM) = multipleMatchingSlateComposer.sardBound(block, dualsArray.value,
        matchingConstraintsPerIndex)
      (nV, 1, corral, Max(corral), jM)
    }.reduce(aggregator.plus(_, _))
    println(f"percent_vertex_soln: ${100.0 * (numI - nonVertexSoln) / numI}\t" +
      f"avg_corral_size: ${corralSize / numI}\t" +
      f"max_corral_size:${corralSizeMax.get}")
    0.5 * nonVertexSoln * (1 - 1.0 / jMax.get)
  }

  /**
    * Convert slates (primal solution) into sufficient statistics of the solution.
    *
    * @param duals
    * @return
    */
  override def getPrimalStats(duals: BSV[Double]): Dataset[PartialPrimalStats] = {
    getPrimal(duals).flatMap { case (_, slates) =>
      slates.map { slate =>
        PartialPrimalStats(slate.costs.toArray, slate.objective, slate.x * slate.x)
      }
    }
  }

  /**
    * Get the primal value for a given dual variable. For matching solver primal is a dataset of slates
    *
    * @param dualsBSV
    * @return dataset of slates
    */
  def getPrimal(dualsBSV: BSV[Double]): Dataset[(String, Seq[Slate])] = {
    val dualsArray: Array[Double] = dualsBSV.toArray
    problemDesign.map(block =>
      (block.id, multipleMatchingSlateComposer.getSlate(block, dualsArray, matchingConstraintsPerIndex)))
  }

  /**
    * Get primal for saving. The schema is simplified for readability of clients:
    * some fields are dropped and some renamed:
    * {
    * blockId: String, // often corresponds to impression in matching problems
    * variables: Array[
    * {
    * item: Int // item id in the variable.
    * value: Double // the value of the variable in primal solution, can be fractional
    * // in matching problems we usually expect variables in a block to
    * // sum to 1.0. More than one non-zero variable can have probabilistic
    * // allocation interpretation.
    * }
    * ]
    * }
    *
    * @param duals
    * @return Optionally the DataFrame with primal solution. None if the functionality is not supported.
    */
  override def getPrimalForSaving(duals: BSV[Double]): Option[DataFrame] = {
    val renamedSchema = "array<struct<item:int,value:double>>"

    val primal = getPrimal(duals).map { case (blockId, slates) =>
      val variables = slates.map { s =>
        (s.costs.toList.head._1 / matchingConstraintsPerIndex, s.x)
      }
      (blockId, variables)
    }.toDF("blockId", "variables")
      .withColumn("variables", col("variables").cast(renamedSchema))
    Option(primal)
  }
}

/**
  * Companion object to load objective function from HDFS
  */
object MultipleMatchingSolverDualObjectiveFunction extends DualPrimalObjectiveLoader {
  /**
    * Load the problem objective and constraints
    *
    * @param inputPaths
    * @param spark
    * @return
    */
  def loadData(inputPaths: InputPaths)(implicit spark: SparkSession):
  (Dataset[MultipleMatchingData], BSV[Double], Int) = {

    // the budget data has three columns
    // entityIndex: the index of the entity
    // constraintIndex: index for the constraint; for a multiple-matching problem with three matching constraints per
    // entity, this column may assume values 0, 1 and 2
    // budgetValue: value of the budget
    val budgetDF = IOUtility.readDataFrame(inputPaths.vectorBPath, inputPaths.format)
      .toDF("entityIndex", "constraintIndex", "budgetValue")

    val matchingConstraintsPerIndex = budgetDF.select("constraintIndex").distinct().count().toInt
    val reindexBudgetDF = budgetDF
      .withColumn("linearIndex", lit(matchingConstraintsPerIndex) * col("entityIndex")
        + col("constraintIndex"))
      .select("linearIndex", "budgetValue")

    val indices = reindexBudgetDF.select("linearIndex").rdd.map(r => r.getInt(0)).collect()
    val values = reindexBudgetDF.select("budgetValue").rdd.map(r => r.getDouble(0)).collect()
    val budgetBSV = new BSV(indices, values, values.length)

    var blocks = IOUtility.readDataFrame(inputPaths.ACblocksPath, inputPaths.format)
    // Make the optional fields of the DataBlock null in the dataframe
    MultipleMatchingData.optionalFields.foreach {
      field =>
        if (Try(blocks(field)).isFailure) {
          blocks = blocks.withColumn(field, lit(null))
        }
    }
    val data = blocks.as[MultipleMatchingData]
      .repartition(spark.sparkContext.defaultParallelism)
      .persist(StorageLevel.MEMORY_ONLY)

    (data, budgetBSV, matchingConstraintsPerIndex)
  }

  /**
    * Code to initialize slate optimizer.
    *
    * @param gamma          - gamma regularization (some optimizers require it)
    * @param slateSize      - slate size
    * @param projectionType - one of available projections (simplex, unitbox, et.c.)
    * @return
    */
  def multipleMatchingSlateComposerChooser(gamma: Double, slateSize: Int, projectionType: ProjectionType):
  MultipleMatchingSlateComposer = {
    projectionType match {
      case Simplex =>
        require(slateSize == 1, "Single slot simplex algorithm requires matching.slateSize = 1")
        require(gamma > 0, "Gamma should be > 0 for simplex algorithm")
        new MultipleMatchingSlateComposer(gamma, new SimplexProjection())
      case SimplexInequality =>
        require(slateSize == 1, "Single slot inequality simplex algorithm requires matching.slateSize = 1")
        require(gamma > 0, "Gamma should be > 0 for simplex algorithm")
        new MultipleMatchingSlateComposer(gamma, new SimplexProjection(inequality = true))
      case BoxCut =>
        require(slateSize == 1, "Single slot box cut algorithm requires matching.slateSize = 1")
        require(gamma > 0, "Gamma should be > 0 for box cut algorithm")
        new MultipleMatchingSlateComposer(gamma, new BoxCutProjection(maxIter = 100, inequality = false))
      case BoxCutInequality =>
        require(slateSize == 1, "Single slot box cut inequality algorithm requires matching.slateSize = 1")
        require(gamma > 0, "Gamma should be > 0 for box cut algorithm")
        new MultipleMatchingSlateComposer(gamma, new BoxCutProjection(maxIter = 100, inequality = true))
      case UnitBox =>
        require(slateSize == 1, "Single slot unit box algorithm requires matching.slateSize = 1")
        require(gamma > 0, "Gamma should be > 0 for unit box projection algorithm")
        new MultipleMatchingSlateComposer(gamma, new UnitBoxProjection())
      case Greedy =>
        require(gamma == 0, "Gamma should be zero for max element slate optimizer")
        require(slateSize == 1, "Single slot algorithm requires matching.slateSize = 1")
        new MultipleMatchingSlateComposer(gamma, new GreedyProjection())
    }
  }

  /**
    * objective loader that conforms to a generic loader API
    *
    * @param gamma          - currently used by all objectives, @todo think about making gamma a trait.
    * @param projectionType - the type of projection used for simple constraints
    * @param args           - custom args that are parsed by the loader
    * @param spark          - spark session
    * @return
    */
  override def apply(gamma: Double, projectionType: ProjectionType, args: Array[String])(implicit spark: SparkSession):
  DualPrimalObjective = {
    val inputPaths = InputPathParamsParser.parseArgs(args)
    val multipleMatchingParams = MatchingParamsParser.parseArgs(args)
    val (data, budget, matchingConstraintsPerIndex) = loadData(inputPaths)
    new MultipleMatchingSolverDualObjectiveFunction(
      data,
      budget,
      matchingConstraintsPerIndex,
      multipleMatchingSlateComposerChooser(gamma, multipleMatchingParams.slateSize, projectionType),
      gamma,
      multipleMatchingParams.enableHighDimOptimization,
      multipleMatchingParams.numLambdaPartitions)
  }
}
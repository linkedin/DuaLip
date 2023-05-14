package com.linkedin.dualip.problem

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.data.MatchingData
import com.linkedin.dualip.objective.distributedobjective.DistributedRegularizedObjective
import com.linkedin.dualip.objective.{DualPrimalObjectiveLoader, PartialPrimalStats}
import com.linkedin.dualip.projection.{BoxCutProjection, GreedyProjection, SimplexProjection, UnitBoxProjection}
import com.linkedin.dualip.slate.{SecondPriceAuctionSlateComposer, SingleSlotComposer, Slate, SlateComposer}
import com.linkedin.dualip.util.ProjectionType._
import com.linkedin.dualip.util.VectorOperations.toBSV
import com.linkedin.dualip.util.{IOUtility, InputPathParamsParser, InputPaths}
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
  *
  * @param problemDesign             - parallelized problem representation
  * @param b                         - constraints vector
  * @param slateComposer             - algorithm to generate primal given dual
  * @param gamma                     - behaves like a regularizer and controls the smoothness of the objective
  * @param enableHighDimOptimization - passthrough parameter to the parent class (spark optimization for very high dimensional problems)
  * @param numLambdaPartitions       - used when enableHighDimOptimization=true, dense lambda vectors coming from executors are partitioned
  *                                  for aggregation. The number of partitions should depend on aggregation parallelism and the dimensionality
  *                                  of lambda. A good rule of thumb is to use a multiple of aggregation parallelism to ensure even load
  *                                  but not too high to keep individual partition sizes large (e.g. 1000) for efficiency:
  *                                  numLambdaPartitions \in [5*parallelism, dim(lambda)/1000]
  * @param spark
  */
class MatchingSolverDualObjectiveFunction(
  problemDesign: Dataset[MatchingData],
  b: BSV[Double],
  slateComposer: SlateComposer,
  gamma: Double,
  enableHighDimOptimization: Boolean,
  numLambdaPartitions: Option[Int]
)(implicit spark: SparkSession) extends DistributedRegularizedObjective(b, gamma, enableHighDimOptimization, numLambdaPartitions) with Serializable {

  import spark.implicits._

  lazy val upperBound: Double = slateComposer match {
    case singleSlotComposer: SingleSlotComposer => singleSlotComposer.getProjection match {
      case _: SimplexProjection => problemDesign.map(_.data.map { case (_, c, _) => c }.max + gamma / 2).reduce(_ + _)
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
      val (nV, corral, jM) = slateComposer.sardBound(block, lambdaArray.value)
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
    * @param lambda
    * @return
    */
  override def getPrimalStats(lambda: BSV[Double]): Dataset[PartialPrimalStats] = {
    getPrimal(lambda).flatMap { case (_, slates) =>
      slates.map { slate =>
        PartialPrimalStats(slate.costs.toArray, slate.objective, slate.x * slate.x)
      }
    }
  }

  /**
    * Get primal for saving. The schema is simplified for readability of clients:
    * some fields are dropped and some renamed:
    * {
    * blockId: String, // often corresponds to impression in matching problems
    * variables: Array[
    * {
    * value: Double // the value of the variable in primal solution, can be fractional
    * // in matching problems we usually expect variables in a block to
    * // sum to 1.0. More than one non-zero variable can have probabilistic
    * // allocation interpretation.
    * items: Array[Int] // item ids in the variable. Often a single element if
    * // we select one item per request. But may be a ranked list of items
    * // if we need to fill a multi-slot slate.
    * }
    * ]
    * }
    * Note. There is a potential optimization to use last primal computed during the optimization.
    * Unlikely to help a lot - cost is equivalent to one extra iteration.
    * TODO: consider case class to define return DataFrame schema
    *
    * @param lambda
    * @return Optionally the DataFrame with primal solution. None if the functionality is not supported.
    */
  override def getPrimalForSaving(lambda: BSV[Double]): Option[DataFrame] = {
    val renamedSchema = "array<struct<value:double,items:array<int>>>"

    val primal = getPrimal(lambda).map { case (blockId, slates) =>
      val variables = slates.map { s =>
        val items = s.costs.map { case (itemId, _) => itemId }
        (s.x, items)
      }
      (blockId, variables)
    }.toDF("blockId", "variables")
      .withColumn("variables", col("variables").cast(renamedSchema))
    Option(primal)
  }

  /**
    * Get the primal value for a given dual variable. For matching solver primal is a dataset of slates
    *
    * @param lambda
    * @return dataset of slates
    */
  def getPrimal(lambda: BSV[Double]): Dataset[(String, Seq[Slate])] = {
    val lambdaArray: Array[Double] = lambda.toArray // for performance
    problemDesign.map(block =>
      (block.id, slateComposer.getSlate(block, lambdaArray)))
  }
}

/**
  * Special parameters only for Matching optimizer
  *
  * @param slateSize                 - number of items selected for each request
  * @param enableHighDimOptimization - spark optimization parameter for gradient computation
  *                                  set to true for very high dimensional lambdas (maybe >100K or 1M).
  *                                  and if each iteration is too slow or driver crashes.
  *                                  Default value is false
  * @param numLambdaPartitions       - number of partitions for lambda vector used in gradient aggregation
  */
case class MatchingSolverParams(slateSize: Int = 1, enableHighDimOptimization: Boolean = false, numLambdaPartitions: Option[Int] = None)

/**
  * Companion object to load objective function from HDFS
  */
object MatchingSolverDualObjectiveFunction extends DualPrimalObjectiveLoader {
  /**
    * Load the problem objective and constraints
    *
    * @param inputPaths
    * @param spark
    * @return
    */
  def loadData(inputPaths: InputPaths)(implicit spark: SparkSession): (Dataset[MatchingData], BSV[Double]) = {
    val budget = IOUtility.readDataFrame(inputPaths.vectorBPath, inputPaths.format)
      .map { case Row(_c0: Number, _c1: Number) => (_c0.intValue(), _c1.doubleValue()) }
      .collect

    val itemIds = budget.toMap.keySet
    // Check if every item has budget information encoded.
    budget.indices.foreach { i: Int =>
      require(itemIds.contains(i), f"$i index does not have a specified constraint")
    }

    val b = toBSV(budget, budget.length)

    var blocks = IOUtility.readDataFrame(inputPaths.ACblocksPath, inputPaths.format)
    // Make the optional fields of the DataBlock null in the dataframe
    MatchingData.optionalFields.foreach {
      field =>
        if (Try(blocks(field)).isFailure) {
          blocks = blocks.withColumn(field, lit(null))
        }
    }
    val data = blocks.withColumnRenamed("memberId", "id")
      .as[MatchingData]
      .repartition(spark.sparkContext.defaultParallelism)
      .persist(StorageLevel.MEMORY_ONLY)

    (data, b)
  }

  /**
    * Code to initialize slate optimizer. Currently the available slate optimizers are hardcoded,
    * consider an option to provide custom optimizer
    *
    * @param gamma          - gamma regularization (some optimizers require it)
    * @param slateSize      - slate size
    * @param projectionType - one of available projections (simplex, unitbox, et.c.)
    * @return
    */
  def slateComposerChooser(gamma: Double, slateSize: Int, projectionType: ProjectionType): SlateComposer = {
    projectionType match {
      case Simplex =>
        require(slateSize == 1, "Single slot simplex algorithm requires matching.slateSize = 1")
        require(gamma > 0, "Gamma should be > 0 for simplex algorithm")
        new SingleSlotComposer(gamma, new SimplexProjection())
      case SimplexInequality =>
        require(slateSize == 1, "Single slot inequality simplex algorithm requires matching.slateSize = 1")
        require(gamma > 0, "Gamma should be > 0 for simplex algorithm")
        new SingleSlotComposer(gamma, new SimplexProjection(inequality = true))
      case BoxCut =>
        require(slateSize == 1, "Single slot box cut algorithm requires matching.slateSize = 1")
        require(gamma > 0, "Gamma should be > 0 for box cutx algorithm")
        new SingleSlotComposer(gamma, new BoxCutProjection(100, inequality = false))
      case BoxCutInequality =>
        require(slateSize == 1, "Single slot box cut algorithm requires matching.slateSize = 1")
        require(gamma > 0, "Gamma should be > 0 for box cut algorithm")
        new SingleSlotComposer(gamma, new BoxCutProjection(100, inequality = true))
      case UnitBox =>
        require(slateSize == 1, "Single slot unit box algorithm requires matching.slateSize = 1")
        require(gamma > 0, "Gamma should be > 0 for unit box projection algorithm")
        new SingleSlotComposer(gamma, new UnitBoxProjection())
      case Greedy =>
        require(gamma == 0, "Gamma should be zero for max element slate optimizer")
        require(slateSize == 1, "Single slot algorithm requires matching.slateSize = 1")
        new SingleSlotComposer(gamma, new GreedyProjection())
      case SecondPrice =>
        require(gamma == 0, "Gamma should be zero for second price slate optimizer")
        require(slateSize >= 1, "Slate size should be >=1")
        new SecondPriceAuctionSlateComposer(slateSize)
    }
  }

  /**
    * objective loader that conforms to a generic loader API
    *
    * @param gamma
    * @param args
    * @param spark
    * @return
    */
  override def apply(gamma: Double, projectionType: ProjectionType, args: Array[String])(implicit spark: SparkSession): DualPrimalObjective = {
    val inputPaths = InputPathParamsParser.parseArgs(args)
    val matchingParams = MatchingParamsParser.parseArgs(args)
    val (data, b) = loadData(inputPaths)
    val slateComposer: SlateComposer = slateComposerChooser(gamma, matchingParams.slateSize, projectionType)
    new MatchingSolverDualObjectiveFunction(data, b, slateComposer, gamma, matchingParams.enableHighDimOptimization, matchingParams.numLambdaPartitions)
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
      opt[Int]("matching.numLambdaPartitions") optional() action { (x, c) => c.copy(numLambdaPartitions = Option(x)) }
    }
    parser.parse(args, MatchingSolverParams()) match {
      case Some(params) => params
      case _ => throw new IllegalArgumentException(s"Parsing the command line arguments ${args.mkString(", ")} failed")
    }
  }
}
package com.linkedin.dualip.problem

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.projection._
import com.linkedin.dualip.slate._
import com.linkedin.dualip.solver.{DistributedRegularizedObjective, PartialPrimalStats}
import com.linkedin.dualip.util.DataFormat.DataFormat
import com.linkedin.dualip.util.ProjectionType.{BoxCut, _}
import com.linkedin.dualip.util.SolverUtility.toBSV
import com.linkedin.dualip.util.{ProjectionType => _, _}
import com.twitter.algebird.{Max, Tuple5Semigroup}
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

import scala.util.Try

/**
 * class for constrained matching solver
 *
 * @param problemDesign                     : dataset for the constrained matching solver
 * @param budget                            : budget vectors for the constrained matching solver
 * @param constrainedMatchingSlateOptimizer : optimizer for the constrained matching problem
 * @param gamma                             : weight for the squared term
 * @param enableHighDimOptimization         : flag to enable high dimensional optimization
 * @param numLambdaPartitions               : number of partitions for the duals
 * @param spark                             : spark session
 */
class ConstrainedMatchingSolverDualObjective(
  problemDesign: Dataset[ConstrainedMatchingDataBlock],
  budget: ConstrainedMatchingBudget,
  constrainedMatchingSlateOptimizer: ConstrainedMatchingSlateOptimizer,
  gamma: Double,
  enableHighDimOptimization: Boolean,
  numLambdaPartitions: Option[Int]
)(implicit spark: SparkSession) extends
  DistributedRegularizedObjective(BSV(budget.budgetLocal.toArray ++ budget.budgetGlobal.toArray), gamma,
    enableHighDimOptimization, numLambdaPartitions) with Serializable {

  import spark.implicits._

  override def getPrimalUpperBound: Double = Double.PositiveInfinity

  /**
   *
   * @param dualsBSV : duals in the form of an object of ConstrainedMatchingDualsBSV
   * @return
   */
  def getSardBound(dualsBSV: ConstrainedMatchingDualsBSV): Double = {
    val lambdaLocal: Array[Double] = spark.sparkContext.broadcast(dualsBSV.lambdaLocal.toArray).value
    val lambdaGlobal: Array[Double] = spark.sparkContext.broadcast(dualsBSV.lambdaGlobal.toArray).value

    val aggregator = new Tuple5Semigroup[Int, Int, Double, Max[Double], Max[Int]]
    val (nonVertexSoln, numI, corralSize, corralSizeMax, jMax) = problemDesign.map { block =>
      val (nV, corral, jM) = constrainedMatchingSlateOptimizer.sardBound(block, ConstrainedMatchingDuals(lambdaLocal, lambdaGlobal))
      (nV, 1, corral, Max(corral), jM)
    }.reduce(aggregator.plus(_, _))
    println(f"percent_vertex_soln: ${100.0 * (numI - nonVertexSoln) / numI}\t" +
      f"avg_corral_size: ${corralSize / numI}\t" +
      f"max_corral_size:${corralSizeMax.get}")
    0.5 * nonVertexSoln * (1 - 1.0 / jMax.get)
  }

  /**
   * gets the primal value for the duals provided
   *
   * @param dualsBSV : duals in the form of an object of ConstrainedMatchingDualsBSV
   * @return
   */
  def getPrimal(dualsBSV: ConstrainedMatchingDualsBSV): Dataset[(String, Seq[Slate])] = {
    val duals = ConstrainedMatchingDuals(dualsBSV.lambdaLocal.toArray, dualsBSV.lambdaGlobal.toArray)
    problemDesign.map { block => {
      (block.id, constrainedMatchingSlateOptimizer.optimize(block, duals))
    }
    }
  }

  /**
   * converts slates (primal solution) into sufficient statistics of the solution
   *
   * @param dualsBSV : duals in the form of an object of ConstrainedMatchingDualsBSV
   * @return
   */
  override def getPrimalStats(lambda: BSV[Double]): Dataset[PartialPrimalStats] = {
    val dualsBSV = ConstrainedMatchingDualsBSV(BSV(lambda.data.take(budget.budgetLocal.length)),
      BSV(lambda.data.takeRight(budget.budgetGlobal.length)))
    getPrimal(dualsBSV).flatMap { case (_, slates) =>
      slates.map { slate => PartialPrimalStats(slate.costs.toArray, slate.objective, slate.x * slate.x) }
    }
  }

  /**
   * gets the primal values for further caching and evaluation
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
   *
   * @param dualsBSV : duals in the form of an object of ConstrainedMatchingDualsBSV
   * @return
   */
  def getPrimalForSaving(dualsBSV: ConstrainedMatchingDualsBSV): Option[DataFrame] = {
    val renamedSchema = "array<struct<value:double,items:array<int>>>"
    val primal = getPrimal(dualsBSV).map { case (blockId, slates) =>
      val variables = slates.map { s =>
        val items = s.costs.map { case (itemId, _) => itemId }
        (s.x, items)
      }
      (blockId, variables)
    }.toDF("blockId", "variables")
      .withColumn("variables", col("variables").cast(renamedSchema))

    Option(primal)
  }
}

object ConstrainedMatchingSolverDualObjective {

  /**
   * checks for consistency of the budget vectors
   *
   * @param budget : array of tuples of the form (Int, Double)
   */
  def checkBudget(budget: Array[(Int, Double)]): Unit = {
    val entityIds = budget.toMap.keySet
    budget.indices.foreach { i: Int =>
      require(entityIds.contains(i), f"$i index does not have a specified constraint")
    }
  }

  /**
   * loads budget vectors
   *
   * @param localBudgetPath  : Path of the budgets corresponding to local constraints
   * @param globalBudgetPath : Path of the budgets corresponding to global constraints
   * @param format           : The format of input data, e.g. avro or orc
   * @param spark            : spark session
   * @return
   */
  def loadBudgetData(localBudgetPath: String, globalBudgetPath: String, format: DataFormat)
    (implicit spark: SparkSession): ConstrainedMatchingBudget = {
    import spark.implicits._
    val budgetLocal = IOUtility.readDataFrame(localBudgetPath, format)
      .map { case Row(_c0: Number, _c1: Number) => (_c0.intValue(), _c1.doubleValue()) }
      .collect

    val budgetGlobal = IOUtility.readDataFrame(globalBudgetPath, format)
      .map { case Row(_c0: Number, _c1: Number) => (_c0.intValue(), _c1.doubleValue()) }
      .collect

    checkBudget(budgetLocal)
    checkBudget(budgetGlobal)
    ConstrainedMatchingBudget(toBSV(budgetLocal, budgetLocal.length), toBSV(budgetGlobal, budgetGlobal.length))
  }

  /**
   * loads A matrix, G matrix and c vector combined in the ConstrainedMatchingDataBlock
   *
   * @param constrainedMatchingDataPath : Path of A matrix, G matrix and c vector combined in a special data block
   * @param numOfPartitions             : number of partitions
   * @param format                      : The format of input data, e.g. avro or orc
   * @param spark                       : spark session
   * @return
   */
  def loadConstrainedMatchingData(constrainedMatchingDataPath: String, numOfPartitions: Int, format: DataFormat)
    (implicit spark: SparkSession): Dataset[ConstrainedMatchingDataBlock] = {
    import spark.implicits._
    var constrainedMatchingDataBlocks = IOUtility.readDataFrame(constrainedMatchingDataPath, format)
      .toDF("id", "data")
    DataBlock.optionalFields.foreach {
      field =>
        if (Try(constrainedMatchingDataBlocks(field)).isFailure) {
          constrainedMatchingDataBlocks = constrainedMatchingDataBlocks.withColumn(field, lit(null))
        }
    }

    constrainedMatchingDataBlocks
      .as[ConstrainedMatchingDataBlock]
      .repartition(numOfPartitions)
      .persist(StorageLevel.MEMORY_ONLY)
  }

  /**
   * loads data for constrained matching problem
   *
   * @param constrainedMatchingDataPath : Path of A matrix, G matrix and c vector combined in a special data block
   * @param localBudgetPath             : Path of the budgets corresponding to local constraints
   * @param globalBudgetPath            : Path of the budgets corresponding to global constraints
   * @param numOfPartitions             : number of partitions
   * @param format                      : The format of input data, e.g. avro or orc
   * @param spark                       : spark session
   * @return
   */
  def loadData(constrainedMatchingDataPath: String, localBudgetPath: String,
    globalBudgetPath: String, numOfPartitions: Int, format: DataFormat)
    (implicit spark: SparkSession): (Dataset[ConstrainedMatchingDataBlock],
    ConstrainedMatchingBudget) = {
    (loadConstrainedMatchingData(constrainedMatchingDataPath, numOfPartitions, format),
      loadBudgetData(localBudgetPath, globalBudgetPath, format))
  }

  /**
   * selects appropriate slate optimizer
   *
   * @param gamma          : weight of the square term
   * @param projectionType : type of projection
   * @return
   */
  def constrainedMatchingSlateOptimizerChooser(gamma: Double, projectionType: ProjectionType):
  ConstrainedMatchingSlateOptimizer = {
    projectionType match {
      case BoxCut =>
        require(gamma > 0, "Gamma should be > 0 for box cut algorithm")
        new ConstrainedMatchingSlateOptimizer(gamma, new BoxCutProjection(100, inequality = true))
    }
  }

  /**
   * objective loader that conforms to a generic loader API
   *
   * @param gamma          : weight of the square term
   * @param projectionType : type of projection
   * @param args           : arguments for constrained matching solver
   * @param spark          : spark session
   * @return
   */
  def apply(gamma: Double, projectionType: ProjectionType, args: Array[String])(implicit spark: SparkSession):
  ConstrainedMatchingSolverDualObjective = {
    val constrainedMatchingParams = ConstrainedMatchingParamsParser.parseArgs(args)
    val (problemDesign, budget) = loadData(constrainedMatchingParams.constrainedMatchingDataPath,
      constrainedMatchingParams.localBudgetPath, constrainedMatchingParams.globalBudgetPath,
      constrainedMatchingParams.numOfPartitions, constrainedMatchingParams.format)
    new ConstrainedMatchingSolverDualObjective(problemDesign, budget,
      constrainedMatchingSlateOptimizerChooser(gamma, projectionType),
      gamma, constrainedMatchingParams.enableHighDimOptimization, constrainedMatchingParams.numLambdaPartitions)
  }
}

/**
 * parser for parameters of the constrained matching problem
 */
object ConstrainedMatchingParamsParser {
  def parseArgs(args: Array[String]): ConstrainedMatchingSolverParams = {
    val parser = new scopt.OptionParser[ConstrainedMatchingSolverParams]("parameter parser for " +
      "constrained matching slate solver") {
      override def errorOnUnknownArgument = false

      opt[String]("input.constrainedMatchingDataPath") required() action { (x, c) =>
        c.copy(constrainedMatchingDataPath = x)
      }
      opt[String]("input.localBudgetPath") required() action { (x, c) => c.copy(localBudgetPath = x) }
      opt[String]("input.globalBudgetPath") required() action { (x, c) => c.copy(globalBudgetPath = x) }
      opt[String]("input.format") required() action { (x, c) => c.copy(format = DataFormat.withName(x)) }
      opt[Int]("input.numOfPartitions") optional() action { (x, c) => c.copy(numOfPartitions = x) }
      opt[Boolean]("constrainedmatching.enableHighDimOptimization") optional() action { (x, c) => c.copy(enableHighDimOptimization = x) }
      opt[Int]("constrainedmatching.numLambdaPartitions") optional() action { (x, c) => c.copy(numLambdaPartitions = Option(x)) }
    }

    parser.parse(args, ConstrainedMatchingSolverParams("", "", "", DataFormat.AVRO, 100)) match {
      case Some(params) => params
      case _ => throw new IllegalArgumentException(s"Parsing the command line arguments ${args.mkString(", ")} failed")
    }
  }
}

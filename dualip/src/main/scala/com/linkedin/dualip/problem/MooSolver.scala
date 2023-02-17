package com.linkedin.dualip.problem

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.problem.MatchingSolverDualObjectiveFunction.toBSV
import com.linkedin.dualip.projection.{BoxCutProjection, Projection, SimplexProjection, UnitBoxProjection}
import com.linkedin.dualip.solver._
import com.linkedin.dualip.util.{IOUtility, InputPathParamsParser, InputPaths, MapReduceArray, MapReduceCollectionWrapper, MapReduceDataset}
import com.linkedin.dualip.util.ProjectionType._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Encoders, Row, SparkSession}
import org.apache.spark.sql.functions.lit
import org.apache.spark.storage.StorageLevel
import scala.math.max

/**
 * A MOO block of data. A vertical slice of design matrix, specifically the variables in the same simplex constraint sum x <= 1
 * We need to keep this data together because we need to do a simplex projection on it.
 *
 * Column (variable indices in "a" and "c" are relative, that is, variable is uniquely identified by
 * a combination of block id and internal id.
 *
 * internal representation is optimized for the operations that algorithm implements and data characteristics:
 * in particular, dense constraints matrix with few rows.
 *
 * @param id - unique identifier of the block, i.e. impression id for some problems
 * @param a - a dense constraints matrix a(row)(column)
 * @param c - a dense objective function vector
 * @param problemId - unique identifier for distinguishing a specific LP problem
 */
case class MooDataBlock(id: Long, a: Array[Array[Double]], c: Array[Double], problemId: Long)

/**
 * A constraint block of data. A data point of constraint vector.
 *
 * @param row - a specific row number of the constraint vector
 * @param value - the corresponding constraint value for the row
 * @param problemId - unique identifier for distinguishing a specific LP problem
 */
case class ConstraintBlock(row: Int, value: Double, problemId: Long)

/**
 * Moo objective function, encapsulates problem design
 * @param problemDesign - constraints matrix and objective vector
 * @param b - constraints vector
 * @param gamma - regularization parameter
 * @param projectionType - type of projection used
 * @param boxCutUpperBound - upper bound for the box cut projection constraint
 * @param spark - implicit spark session object
 */
class MooSolverDualObjectiveFunction(
  problemDesign: MapReduceCollectionWrapper[MooDataBlock],
  b: BSV[Double],
  gamma: Double,
  projectionType: ProjectionType,
  boxCutUpperBound: Int = 1
)(implicit spark: SparkSession) extends MooDistributedRegularizedObjective(b, gamma) with Serializable {

  // The simple constraints are encoded using the projections supported by the slate optimizer type
  lazy val projection: Projection = projectionType match {
    case Simplex => new SimplexProjection(checkVertexSolution = true)
    case SimplexInequality => new SimplexProjection(checkVertexSolution = true, inequality = true)
    case BoxCut => new BoxCutProjection(maxIter = boxCutUpperBound)
    case BoxCutInequality => new BoxCutProjection(maxIter = boxCutUpperBound, inequality = true)
    case UnitBox => new UnitBoxProjection()
    case _ => throw new NoClassDefFoundError(s"Projection $projection is not supported by MOOSolver.")
  }

  lazy val upperBound: Double = projectionType match {
    case Simplex => problemDesign.map(_.c.max + gamma / 2, MooSolverDualObjectiveFunction.encoderDouble).reduce(_ + _)
    case SimplexInequality => problemDesign.map(problem => max(0, problem.c.max) + gamma / 2, MooSolverDualObjectiveFunction.encoderDouble).reduce(_ + _)
    case _ => Double.PositiveInfinity
  }

  override def getPrimalUpperBound: Double = upperBound

  /**
   * Method called in the parent class
   * @param lambda the dual variable
   * @return
   */
  override def getPrimalStats(lambda: BSV[Double]): MapReduceCollectionWrapper[PartialPrimalStats] = {
    val primals = getPrimal(lambda)
    primals.map( { case (_, _, stats) => stats } , MooSolverDualObjectiveFunction.encoderPartialPrimalStats)
  }

  /**
   * Get the primal value for a given dual variable along with some auxiliary quantities needed for
   * the solver.
   * This is the most expensive part of the algorithm, so we pay attention to code optimization
   * and use java arrays, while loops and mutable variables.
   * @param lambda the dual variable
   * @return (id, obj, sum(`x^2`), grad)
   */
  def getPrimal(lambda: BSV[Double]): MapReduceCollectionWrapper[(Long, Array[Double], PartialPrimalStats)] = {
    val lambdaArray = lambda.toArray
    val metadata: Map[String, Double] = Map[String, Double]("boxCut" -> boxCutUpperBound)
    problemDesign.map( { block =>
      // compute projection input
      val n = block.c.length // number of columns (variables) in block
      val m = block.a.length // number of rows (constraints) in block
      val vectorForProjection = new Array[Double](n) // (- c - lambda * A)/gamma
      var i = 0
      var j = 0
      while (i < n) {
        vectorForProjection(i) = -1.0 * block.c(i)
        j = 0
        while (j < m) {
          vectorForProjection(i) -= block.a(j)(i) * lambdaArray(j)
          j += 1
        }
        vectorForProjection(i) /= gamma
        i += 1
      }
      // get the projection (the primal solution)
      val primal = projection.project(BSV(vectorForProjection), metadata).toArray

      // compute cx, xx, ax values for the given primal
      var obj = 0.0 // cx running sum
      var xsquared = 0.0 // xx running sum
      val constr = new Array[Double](m) // ax running sum
      i = 0
      while (i < n) {
        obj += primal(i) * block.c(i)
        xsquared += primal(i) * primal(i)
        j = 0
        while (j < m) {
          constr(j) += block.a(j)(i) * primal(i)
          j += 1
        }
        i += 1
      }
      val ax = constr.zipWithIndex.map { case (value, index) => (index, value) }
      (block.id, primal, PartialPrimalStats(ax, obj, xsquared))
    }, MooSolverDualObjectiveFunction.encoderTuplePartialPrimalStats)
  }

  /**
    * Return the primal for saving as a DataFrame.
    *
    * @param lambda The dual values
    * @return Optionally the DataFrame with primal solution. None if the functionality is not supported.
    */
  override def getPrimalForSaving(lambda: BSV[Double]): Option[DataFrame] = {

    import spark.implicits._

    val primalsDataFrame = getPrimal(lambda).map({ case (id, primal, _) => (id, primal) },
      MooSolverDualObjectiveFunction.encoderIdPrimal)
      .value
      .asInstanceOf[Dataset[(Long, Array[Double])]]
      .map({ case (blockId, primals) => (blockId.toString,
        primals.zipWithIndex.map { case (value, index) => (index, value) })
      })
      .toDF("blockId", "variables")

    Option(primalsDataFrame)
  }
}

/**
 * This companion object encapsulates all data/objective loading specifics for (single) MOO use case
 */
object MooSolverDualObjectiveFunction extends DualPrimalObjectiveLoader {
  /**
   * Create encoder singletons to reuse and prevent re-initialization costs
   */
  val encoderTuplePartialPrimalStats: Encoder[(Long, Array[Double], PartialPrimalStats)] = ExpressionEncoder[(Long, Array[Double], PartialPrimalStats)]
  val encoderPartialPrimalStats: Encoder[PartialPrimalStats] = ExpressionEncoder[PartialPrimalStats]
  val encoderIdPrimal: Encoder[(Long, Array[Double])] = ExpressionEncoder[(Long, Array[Double])]
  val encoderDouble: Encoder[Double] = Encoders.scalaDouble

  val DUMMY_PROBLEM_ID: Long = -1  // this is used when we are solving just a single problem
  /**
   * Custom data loader.
   * @param inputPaths input path for vectorB and ACblock
   * @param spark spark session
   * @return
   */
  def loadData(inputPaths: InputPaths)
    (implicit spark: SparkSession): (MapReduceDataset[MooDataBlock], BSV[Double]) = {
    import spark.implicits._

    val budget = IOUtility.readDataFrame(inputPaths.vectorBPath, inputPaths.format)
      .map{case Row(_c0: Number, _c1: Number) => (_c0.intValue(), _c1.doubleValue()) }
      .toDF("row", "value")
      .withColumn("problemId", lit(DUMMY_PROBLEM_ID))
      .as[ConstraintBlock]
      .map{constraintBlock => (constraintBlock.row, constraintBlock.value) }
      .collect

    val itemIds = budget.toMap.keySet
    // Check if every item has budget information encoded.
    budget.indices.foreach { i: Int =>
      require(itemIds.contains(i), f"$i index does not have a specified constraint" )
    }

    val b = toBSV(budget, budget.length)

    val data = IOUtility.readDataFrame(inputPaths.ACblocksPath, inputPaths.format)
      .repartition(spark.sparkContext.defaultParallelism)
      .withColumn("problemId", lit(DUMMY_PROBLEM_ID))
      .as[MooDataBlock]
      .persist(StorageLevel.MEMORY_ONLY)

    val retData = MapReduceDataset[MooDataBlock](data)

    (retData, b)
  }

  /**
   * objective loader that conforms to a generic loader API
   * @param gamma regularization weight
   * @param projectionType type of projection used
   * @param args input arguments
   * @param spark spark session
   * @return
   */
  def apply(gamma: Double, projectionType: ProjectionType, args: Array[String])(implicit spark: SparkSession): DualPrimalDifferentiableObjective = {
    val inputPaths = InputPathParamsParser.parseArgs(args)

    val (data, b) = MooSolverDualObjectiveFunction.loadData(inputPaths)

    new MooSolverDualObjectiveFunction(data, b, gamma, projectionType)
  }

  /**
    * objective loader with a customized box cut projection upper bound
    * @param gamma regularization weight
    * @param projectionType type of projection used
    * @param boxCutUpperBound upper bound for the box cut projection constraint
    * @param args input arguments
    * @param spark spark session
    * @return
    */
  def applyWithCustomizedBoxCut(gamma: Double, projectionType: ProjectionType, boxCutUpperBound: Int, args: Array[String])
                               (implicit spark: SparkSession): DualPrimalDifferentiableObjective = {
    val inputPaths = InputPathParamsParser.parseArgs(args)

    val (data, b) = MooSolverDualObjectiveFunction.loadData(inputPaths)

    new MooSolverDualObjectiveFunction(data, b, gamma, projectionType, boxCutUpperBound)
  }
}

/**
 * This object encapsulates all data/objective loading specifics for Parallel MOO use case
 */
object ParallelMooSolverDualObjectiveFunction {
  /**
   * Custom data loader.
   *
   * @param inputPaths input path for vectorB and ACblock
   * @param gamma      gamma regularization
   * @param projectionType type of projection used
   * @param spark      spark session
   * @return
   */
  def loadData(inputPaths: InputPaths, gamma: Double, projectionType: ProjectionType)
    (implicit spark: SparkSession): Dataset[(Long, MapReduceArray[MooDataBlock], Array[(Int, Double)])] = {
    import spark.implicits._

    val budgetData = IOUtility.readDataFrame(inputPaths.vectorBPath, inputPaths.format)
      .as[ConstraintBlock]
      .groupByKey(constraintBlock => constraintBlock.problemId)
      .mapGroups { case (problemId, dataIterator) =>
        (problemId, dataIterator.map{constraintBlock => (constraintBlock.row, constraintBlock.value) }.toArray)
      }
      .toDF("problemId", "budget")

    val mooData = IOUtility.readDataFrame(inputPaths.ACblocksPath, inputPaths.format)
      .as[MooDataBlock]
      .groupByKey(mooDataBlock => mooDataBlock.problemId)
      .mapGroups { case (problemId, dataIterator) =>
        (problemId, MapReduceArray[MooDataBlock](dataIterator.toArray))
      }
      .toDF("problemId", "mooData")

    val retData = mooData.join(budgetData, "problemId")
      .select($"problemId", $"mooData", $"budget")
      .as[(Long, MapReduceArray[MooDataBlock], Array[(Int, Double)])]

    retData
  }
}
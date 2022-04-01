package com.linkedin.dualip.problem

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.projection.BoxCutProjection
import com.linkedin.dualip.slate.ConstrainedMatchingSlateOptimizer
import com.linkedin.dualip.solver.{AcceleratedGradientDescent, DualPrimalDifferentiableComputationResult}
import com.linkedin.dualip.util.{ConstrainedMatchingBudget, ConstrainedMatchingDataBlock, ConstrainedMatchingDualsBSV}
import com.linkedin.spark.common.lib.TestUtils
import org.apache.spark.sql.SparkSession
import org.testng.Assert
import org.testng.annotations.Test

import scala.collection.mutable
import scala.util.Random

case class ConstrainedMatchingSolverTestVar(value: Double, items: Array[Int])

case class ConstrainedMatchingSolverTestRow(blockId: String, variables: Array[ConstrainedMatchingSolverTestVar])

class ConstrainedMatchingSolverTest {

  implicit val spark: SparkSession = TestUtils.createSparkSession()

  import spark.implicits._

  spark.sparkContext.setLogLevel("warn")

  def generateRandomArray(arraySize: Int): Array[Double] = {
    val randomObject: Random.type = scala.util.Random
    (for (_ <- 1 to arraySize) yield randomObject.nextFloat().toDouble).toArray
  }

  val enableHighDimOptimization = false
  // unit tests should work with both true/false settings, only using "false" to save runtime

  val numberOfLocalConstraints = 3
  val numberOfGlobalConstraints = 2
  val numberOfBlocks = 4
  val tolerance = 1E-5
  val dualTolerance = 1E-10

  // the following matrix is of dimension numberOfLocalConstraints X numberOfBlocks
  val localConstraintsMat = Map(
    (0, 0) -> 0.20, (1, 0) -> 0.25, (2, 0) -> 0.22,
    (0, 1) -> 0.30, (1, 1) -> 0.35, (2, 1) -> 0.32,
    (0, 2) -> 0.40, (1, 2) -> 0.45, (2, 2) -> 0.42,
    (0, 3) -> 0.18, (1, 3) -> 0.15, (2, 3) -> 0.12
  )

  // the following matrix is of dimension numberOfLocalConstraints X numberOfBlocks
  val costValueMat = Map(
    (0, 0) -> -1.00, (0, 1) -> -1.15, (0, 2) -> -1.80, (0, 3) -> -2.20,
    (1, 0) -> -2.00, (1, 1) -> -1.10, (1, 2) -> -1.30, (1, 3) -> -1.85,
    (2, 0) -> -1.50, (2, 1) -> -2.10, (2, 2) -> -1.20, (2, 3) -> -1.45
  )

  // the following matrix is of dimension numberOfGlobalConstraints X numberOfLocalConstraints X numberOfBlocks
  val globalConstraintsMat = Map(
    (0, 0, 0) -> 0.20, (0, 1, 0) -> 0.20, (0, 2, 0) -> 0.20,
    (0, 0, 1) -> 0.30, (0, 1, 1) -> 0.30, (0, 2, 1) -> 0.30,
    (0, 0, 2) -> 0.40, (0, 1, 2) -> 0.40, (0, 2, 2) -> 0.40,
    (0, 0, 3) -> 0.18, (0, 1, 3) -> 0.18, (0, 2, 3) -> 0.18,
    (1, 0, 0) -> 0.25, (1, 1, 0) -> 0.25, (1, 2, 0) -> 0.25,
    (1, 0, 1) -> 0.35, (1, 1, 1) -> 0.35, (1, 2, 1) -> 0.35,
    (1, 0, 2) -> 0.45, (1, 1, 2) -> 0.45, (1, 2, 2) -> 0.45,
    (1, 0, 3) -> 0.15, (1, 1, 3) -> 0.15, (1, 2, 3) -> 0.15
  )

  val gamma = 0.01
  val metadata: Map[String, Double] = Map[String, Double]("boxCut" -> 5)
  val data: Seq[ConstrainedMatchingDataBlock] = (0 until numberOfBlocks).map(blockIndex =>
    ConstrainedMatchingDataBlock(blockIndex.toString, (0 until numberOfLocalConstraints)
      .map(localConstraintIndex => {
        (localConstraintIndex, costValueMat((localConstraintIndex, blockIndex)),
          localConstraintsMat((localConstraintIndex, blockIndex)), (0 until numberOfGlobalConstraints)
          .map(globalConstraintIndex => (globalConstraintIndex, globalConstraintsMat((globalConstraintIndex,
            localConstraintIndex, blockIndex)))))
      }
      ), metadata))

  // local and global budgets
  val budget: ConstrainedMatchingBudget = ConstrainedMatchingBudget(BSV.fill(numberOfLocalConstraints)(0.30),
    BSV.fill(numberOfGlobalConstraints)(0.90))

  // initial values of the duals
  val initialValueDuals: ConstrainedMatchingDualsBSV = ConstrainedMatchingDualsBSV(BSV.fill(numberOfLocalConstraints)(0.10),
    BSV.fill(numberOfGlobalConstraints)(0.10))

  // expected values for this problem were computed with SCS
  val expectedDualObjective: Double = -8.288214
  val expectedDuals: ConstrainedMatchingDualsBSV = ConstrainedMatchingDualsBSV(BSV(Array(0.0, 3.991200, 2.252802)),
    BSV(Array(2.320209, 2.128598)))

  def checkConsistencyLambda(lambda: BSV[Double], expectedLambda: BSV[Double]): Boolean = {
    (0 until lambda.length).map { i =>
      if (Math.abs(lambda(i) - expectedLambda(i)) < tolerance) true else false
    }.forall(_ == true)
  }

  def checkConsistency(value: DualPrimalDifferentiableComputationResult): Unit = {
    Assert.assertTrue((Math.abs(value.dualObjective - expectedDualObjective) < tolerance) ||
      checkConsistencyLambda(value.lambda, BSV(expectedDuals.lambdaLocal.toArray
        ++ expectedDuals.lambdaGlobal.toArray)))
  }

  @Test
  def testPrimal(): Unit = {
    val constrainedMatchingSlateOptimizer: ConstrainedMatchingSlateOptimizer = new ConstrainedMatchingSlateOptimizer(
      gamma, new BoxCutProjection(1000, inequality = true))
    val f = new ConstrainedMatchingSolverDualObjective(spark.createDataset(data), budget,
      constrainedMatchingSlateOptimizer, gamma, enableHighDimOptimization, None)
    val expectedDualsBSV = BSV(expectedDuals.lambdaLocal.toArray ++ expectedDuals.lambdaGlobal.toArray)

    // compute value using calculate function
    val value = f.calculate(expectedDualsBSV, mutable.Map.empty, 1)

    // alternative computation using returned primal solution and hard-coded objective
    val primal = f.getPrimalForSaving(expectedDualsBSV).get
      .as[ConstrainedMatchingSolverTestRow].collect()
      .flatMap { r =>
        r.variables.map { variables =>
          (r.blockId.toInt, variables.items.min, variables.value)
        }
      }
    val primalObj = primal.map { case (blockIndex, localConstraintIndex, x) => x *
      costValueMat((localConstraintIndex, blockIndex)) + 0.5 * gamma * x * x
    }.sum
    Assert.assertTrue(Math.abs(value.primalObjective - primalObj) < tolerance)
  }

  @Test
  def testBoxCutInequalitySolver(): Unit = {
    val constrainedMatchingSlateOptimizer: ConstrainedMatchingSlateOptimizer = new ConstrainedMatchingSlateOptimizer(
      gamma, new BoxCutProjection(1000, inequality = true))
    val f = new ConstrainedMatchingSolverDualObjective(spark.createDataset(data), budget,
      constrainedMatchingSlateOptimizer, gamma, enableHighDimOptimization, None)

    val optimizer = new AcceleratedGradientDescent(maxIter = 1000, dualTolerance = dualTolerance)
    val (_, value, _) = optimizer.maximize(f, BSV(initialValueDuals.lambdaLocal.toArray ++
      initialValueDuals.lambdaGlobal.toArray), 1)
    checkConsistency(value)
  }
}
package com.linkedin.dualip.problem

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.data.{ConstrainedMatchingBudget, ConstrainedMatchingData, ConstrainedMatchingDuals, ConstrainedMatchingDualsBSV}
import com.linkedin.dualip.maximizer.solver.firstorder.gradientbased.AcceleratedGradientDescent
import com.linkedin.dualip.objective.DualPrimalComputationResult
import com.linkedin.dualip.projection.BoxCutProjection
import com.linkedin.dualip.slate.{ConstrainedMatchingSlateComposer, Slate}
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

  val numberOfLocalConstraints = 2
  val numberOfGlobalConstraints = 1
  val numberOfBlocks = 3
  val tolerance = 1E-5
  val dualTolerance = 1E-10
  val gamma = 0.01

  // the following matrix is of dimension numberOfLocalConstraints X numberOfBlocks
  val localConstraintsMat: Map[(Int, Int), Double] = Map(
    (0, 0) -> 0.20, (0, 1) -> 0.30, (0, 2) -> 0.40,
    (1, 0) -> 0.25, (1, 1) -> 0.35, (1, 2) -> 0.45
  )

  // the following matrix is of dimension numberOfLocalConstraints X numberOfBlocks
  val costValueMat: Map[(Int, Int), Double] = Map(
    (0, 0) -> -10.00, (0, 1) -> -10.15, (0, 2) -> -10.80,
    (1, 0) -> -20.00, (1, 1) -> -10.10, (1, 2) -> -10.30
  )

  // the following matrix is of dimension numberOfGlobalConstraints X numberOfLocalConstraints X numberOfBlocks
  val globalConstraintsMat: Map[(Int, Int, Int), Double] = Map(
    // global constraints for index 0
    (0, 0, 0) -> 1.20, (0, 0, 1) -> 0.0, (0, 0, 2) -> 0.0,
    (0, 1, 0) -> 0.0, (0, 1, 1) -> 1.30, (0, 1, 2) -> 1.40
  )

  val metadata: Map[String, Double] = Map[String, Double]("boxCut" -> 5)
  val data: Seq[ConstrainedMatchingData] = (0 until numberOfBlocks).map(blockIndex =>
    ConstrainedMatchingData(blockIndex.toString, (0 until numberOfLocalConstraints)
      .map(localConstraintIndex => {
        (localConstraintIndex, costValueMat((localConstraintIndex, blockIndex)),
          localConstraintsMat((localConstraintIndex, blockIndex)), (0 until numberOfGlobalConstraints)
          .map(globalConstraintIndex => (globalConstraintIndex, globalConstraintsMat((globalConstraintIndex,
            localConstraintIndex, blockIndex)))))
      }
      ), metadata))

  // local and global budgets
  val budget: ConstrainedMatchingBudget = ConstrainedMatchingBudget(BSV.fill(numberOfLocalConstraints)(0.30),
    BSV.fill(numberOfGlobalConstraints)(2.50))

  // initial values of the duals
  val initialValuesLocalDuals: BSV[Double] = BSV.fill(numberOfLocalConstraints)(0.15)
  val initialValuesGlobalDuals: BSV[Double] = BSV.fill(numberOfGlobalConstraints)(0.25)
  val initialValueDuals: ConstrainedMatchingDualsBSV = ConstrainedMatchingDualsBSV(initialValuesLocalDuals,
    initialValuesGlobalDuals)
  val initialValueDualsBSV: BSV[Double] = BSV(initialValuesGlobalDuals.toArray ++ initialValuesLocalDuals.toArray)

  // expected values for this problem were computed with SCS
  val expectedDualObjective: Double = -32.470503
  val localDuals: BSV[Double] = BSV(Array(3.991200, 2.252802))
  val globalDuals: BSV[Double] = BSV(Array(2.320209))
  val duals: ConstrainedMatchingDualsBSV = ConstrainedMatchingDualsBSV(localDuals, globalDuals)
  val dualsBSV: BSV[Double] = BSV(duals.lambdaGlobal.toArray ++ duals.lambdaLocal.toArray)
  val dualsConstrainedMatching: ConstrainedMatchingDuals = ConstrainedMatchingDuals(localDuals.toArray,
    globalDuals.toArray)

  // slate and objective
  val constrainedMatchingSlateComposer: ConstrainedMatchingSlateComposer = new ConstrainedMatchingSlateComposer(
    gamma, new BoxCutProjection(100, inequality = true))
  val f = new ConstrainedMatchingSolverDualObjectiveFunction(spark.createDataset(data), budget,
    constrainedMatchingSlateComposer, gamma, enableHighDimOptimization, None)

  def checkConsistencyLambda(lambda: BSV[Double], expectedLambda: BSV[Double]): Boolean = {
    (0 until lambda.length).map { i =>
      if (Math.abs(lambda(i) - expectedLambda(i)) < tolerance) true else false
    }.forall(_ == true)
  }

  def checkConsistency(value: DualPrimalComputationResult): Unit = {
    Assert.assertTrue((Math.abs(value.dualObjective - expectedDualObjective) < tolerance) ||
      checkConsistencyLambda(value.lambda, dualsBSV))
  }

  def checkArrayEquality(array: Array[Double], expectedArray: Array[Double]): Boolean = {
    array.indices.map { i =>
      if (Math.abs(array(i) - expectedArray(i)) < tolerance) true else false
    }.forall(_ == true)
  }

  def checkSparseArrayEquality(array: Seq[(Int, Double)], expectedArray: Seq[(Int, Double)]): Boolean = {
    array.indices.map { i =>
      if ((Math.abs(array(i)._2 - expectedArray(i)._2) < tolerance) && (array(i)._1 == expectedArray(i)._1)) true else
        false
    }.forall(_ == true)
  }

  @Test
  def testComputeReducedCosts(): Unit = {
    val expectedCosts = Array(641.75092, 1943.67995)
    val costs = constrainedMatchingSlateComposer.computeReducedCosts(data.head.data.toArray,
      dualsConstrainedMatching)
    Assert.assertTrue(checkArrayEquality(costs, expectedCosts))
  }

  @Test
  def testGetSlate(): Unit = {
    val expectedSlates = Seq(
      Slate(1.0, -10.0, Seq((1, 0.2), (0, 1.20))),
      Slate(1.0, -20.0, Seq((2, 0.25), (0, 0.0)))
    )
    val slates = constrainedMatchingSlateComposer.getSlate(data.head, dualsConstrainedMatching)

    slates.zipWithIndex.foreach { case (slate, index) =>
      Assert.assertTrue(slate.objective == expectedSlates(index).objective)
      Assert.assertTrue(slate.x == expectedSlates(index).x)
      Assert.assertTrue(checkSparseArrayEquality(slate.costs, expectedSlates(index).costs))
    }
  }

  @Test
  def testConvertBSVtoConstrainedMatchingDualsBSV(): Unit = {
    val constrainedMatchingDuals = f.convertBSVtoConstrainedMatchingDualsBSV(dualsBSV)
    Assert.assertTrue(checkConsistencyLambda(constrainedMatchingDuals.lambdaLocal, localDuals))
    Assert.assertTrue(checkConsistencyLambda(constrainedMatchingDuals.lambdaGlobal, globalDuals))
  }

  @Test
  def testGetPrimalForSaving(): Unit = {
    // compute value using calculate function
    val value = f.calculate(dualsBSV, mutable.Map.empty, 1)

    // alternative computation using returned primal solution and hard-coded objective
    val primal = f.getPrimalForSaving(dualsBSV).get
      .as[ConstrainedMatchingSolverTestRow].collect()
      .flatMap { r =>
        r.variables.map { variables =>
          (r.blockId.toInt, variables.items.max, variables.value)
        }
      }
    val primalObj = primal.map { case (blockIndex, localConstraintIndex, x) => x *
      costValueMat((localConstraintIndex - numberOfGlobalConstraints, blockIndex)) + 0.5 * gamma * x * x
    }.sum
    Assert.assertTrue(Math.abs(value.primalObjective - primalObj) < tolerance)
  }

  @Test
  def testBoxCutInequalitySolver(): Unit = {
    val optimizer = new AcceleratedGradientDescent(maxIter = 1000, dualTolerance = dualTolerance)
    val (_, value, _) = optimizer.maximize(f, initialValueDualsBSV, 1)
    Assert.assertTrue(Math.abs(value.dualObjective - expectedDualObjective) < tolerance)
  }
}
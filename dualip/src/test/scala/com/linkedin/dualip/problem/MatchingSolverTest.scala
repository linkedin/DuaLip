package com.linkedin.dualip.problem

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.data.MatchingData
import com.linkedin.dualip.maximizer.solver.firstorder.gradientbased.AcceleratedGradientDescent
import com.linkedin.dualip.projection.{BoxCutProjection, GreedyProjection, SimplexProjection}
import com.linkedin.dualip.slate.{SingleSlotComposer, SlateComposer}
import com.linkedin.spark.common.lib.TestUtils
import org.apache.spark.sql.SparkSession
import org.testng.Assert
import org.testng.annotations.Test

import scala.collection.mutable

// schema for DataFrame in primal test
case class MatchingSolverTestVar(value: Double, items: Array[Int])

case class MatchingSolverTestRow(blockId: String, variables: Array[MatchingSolverTestVar])

class MatchingSolverTest {
  // unit tests should work with both true/false settings, only using "false" to save runtime
  val enableHighDimOptimization = false

  // a(i, j): cost associated with user i for item j
  // c(i, j): objective function coefficient associated with user i for item j
  val a: Map[(Int, Int), Double] = Map(
    (0, 0) -> 0.307766110869125, (0, 1) -> 0.483770735096186, (0, 2) -> 0.624996477039531, (0, 3) -> 0.669021712383255, (0, 4) -> 0.535811153938994,
    (1, 0) -> 0.257672501029447, (1, 1) -> 0.812402617651969, (1, 2) -> 0.882165518123657, (1, 3) -> 0.204612161964178, (1, 4) -> 0.710803845431656,
    (2, 0) -> 0.552322433330119, (2, 1) -> 0.370320537127554, (2, 2) -> 0.28035383997485, (2, 3) -> 0.357524853432551, (2, 4) -> 0.538348698290065,
    (3, 0) -> 0.0563831503968686, (3, 1) -> 0.546558595029637, (3, 2) -> 0.398487901547924, (3, 3) -> 0.359475114848465, (3, 4) -> 0.74897222686559,
    (4, 0) -> 0.468549283919856, (4, 1) -> 0.170262051047757, (4, 2) -> 0.76255108229816, (4, 3) -> 0.690290528349578, (4, 4) -> 0.420101450523362
  )
  val c: Map[(Int, Int), Double] = Map(
    (0, 0) -> -0.307766110869125, (0, 1) -> -0.483770735096186, (0, 2) -> -0.624996477039531, (0, 3) -> -0.669021712383255, (0, 4) -> -0.535811153938994,
    (1, 0) -> -0.257672501029447, (1, 1) -> -0.812402617651969, (1, 2) -> -0.882165518123657, (1, 3) -> -0.204612161964178, (1, 4) -> -0.710803845431656,
    (2, 0) -> -0.552322433330119, (2, 1) -> -0.370320537127554, (2, 2) -> -0.28035383997485, (2, 3) -> -0.357524853432551, (2, 4) -> -0.538348698290065,
    (3, 0) -> -0.0563831503968686, (3, 1) -> -0.546558595029637, (3, 2) -> -0.398487901547924, (3, 3) -> -0.359475114848465, (3, 4) -> -0.74897222686559,
    (4, 0) -> -0.468549283919856, (4, 1) -> -0.170262051047757, (4, 2) -> -0.76255108229816, (4, 3) -> -0.690290528349578, (4, 4) -> -0.420101450523362
  )
  val metadata: Map[String, Double] = Map[String, Double]("boxCut" -> 2)
  val data: Seq[MatchingData] = (0 to 4).map(i =>
    MatchingData(i.toString, (0 to 4).map(j => (j, c((i, j)), a((i, j)))), metadata))
  val b: Array[Double] = Array(0.7, 0.7, 0.7, 0.7, 0.7)

  // Expected values for this problem were computed with SCS
  val expectedDualObjective: Double = -3.4686
  val expectedLambda: Array[Double] = Array(0.0000000, 0.3327713, 0.3855439, 0.3212216, 0.5130992)

  // primal upper bound = \sum_i (max_j c(i, j))
  val expectedPrimalUpperBound: Double = c(0, 0) + c(1, 3) + c(2, 2) + c(3, 0) + c(4, 1)

  @Test
  def testMaxSolver(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    import spark.implicits._
    spark.sparkContext.setLogLevel("warn")

    val gamma = 1E-6
    val slateComposer: SlateComposer = new SingleSlotComposer(gamma, new GreedyProjection())
    val f = new MatchingSolverDualObjectiveFunction(spark.createDataset(data), BSV(b), slateComposer, gamma, enableHighDimOptimization, None)

    val optimizer = new AcceleratedGradientDescent(maxIter = 100)
    val (lambda, value, _) = optimizer.maximize(f, BSV.fill(5)(0.1))
    (0 to 4).foreach { i =>
      Assert.assertTrue(Math.abs(lambda(i) - expectedLambda(i)) < 0.05) // converges closer if we run more iterations
    }
    Assert.assertTrue(Math.abs(value.dualObjective - expectedDualObjective) < 0.05)
  }

  @Test
  def testPrimal(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    import spark.implicits._
    spark.sparkContext.setLogLevel("warn")

    val gamma = 1E-6
    val slateComposer: SlateComposer = new SingleSlotComposer(gamma, new GreedyProjection())
    val f = new MatchingSolverDualObjectiveFunction(spark.createDataset(data), BSV(b), slateComposer, gamma, enableHighDimOptimization, None)
    // compute value using calculate function
    val value = f.calculate(BSV(expectedLambda), mutable.Map.empty, 1)

    // alternative computation using returned primal solution and hardcoded objective
    val primal = f.getPrimalForSaving(BSV(expectedLambda)).get.as[MatchingSolverTestRow].collect.flatMap { r =>
      r.variables.map { x =>
        require(x.items.length == 1)
        (r.blockId.toInt, x.items(0), x.value)
      }
    }
    val primalObj = primal.map { case (i, j, x) => x * c((i, j)) + gamma * x * x / 2.0 }.sum
    Assert.assertTrue(Math.abs(value.primalObjective - primalObj) < 1E-8)
  }

  @Test
  def testSimplexSolver(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    import spark.implicits._
    spark.sparkContext.setLogLevel("warn")

    val gamma = 1E-3
    val slateComposer: SlateComposer = new SingleSlotComposer(gamma, new SimplexProjection())
    val f = new MatchingSolverDualObjectiveFunction(spark.createDataset(data), BSV(b), slateComposer, gamma, enableHighDimOptimization, None)

    val primalUpperBound: Double = expectedPrimalUpperBound + 5 * gamma / 2
    Assert.assertTrue(Math.abs(f.getPrimalUpperBound - primalUpperBound) < 0.01)

    val optimizer = new AcceleratedGradientDescent(maxIter = 200)

    val initialLambda = BSV.fill(5)(0.1)
    val (lambda, value, _) = optimizer.maximize(f, initialLambda)
    (0 to 4).foreach { i =>
      Assert.assertTrue(Math.abs(lambda(i) - expectedLambda(i)) < 0.01)
    }
    Assert.assertTrue(Math.abs(value.dualObjective - expectedDualObjective) < 0.01)
  }

  @Test
  def testBoxCutSolver(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    import spark.implicits._
    spark.sparkContext.setLogLevel("warn")

    val gamma = 1E-6
    val slateComposer: SlateComposer = new SingleSlotComposer(gamma, new BoxCutProjection(1000))
    val f = new MatchingSolverDualObjectiveFunction(spark.createDataset(data), BSV(b), slateComposer, gamma, enableHighDimOptimization, None)

    val optimizer = new AcceleratedGradientDescent(maxIter = 200)

    val initialLambda = BSV.fill(5)(0.1)
    val (lambda, _, _) = optimizer.maximize(f, initialLambda)
    (0 to 4).foreach { i =>
      Assert.assertTrue(Math.abs(lambda(i) - 1.0) < 0.01)
    }
  }

  /**
    * The objective of this test is to simulate the optimal solution of x=0; which tests the correctness of the
    * code-segment that handles the empty primal stats
    */
  @Test
  def testBoxCutInequalitySolver(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    import spark.implicits._

    val data: Seq[MatchingData] = (0 to 4).map(i =>
      MatchingData(i.toString, (0 to 4).map(j => (j, -c((i, j)), a((i, j)))), metadata))
    val b: Array[Double] = Array(0.7, 0.7, 0.7, 0.7, 0.7)
    val gamma = 1e-5

    val slateComposer: SlateComposer = new SingleSlotComposer(gamma, new BoxCutProjection(1000,
      inequality = true))
    val f = new MatchingSolverDualObjectiveFunction(spark.createDataset(data), BSV(b), slateComposer, gamma,
      enableHighDimOptimization, None)

    val optimizer = new AcceleratedGradientDescent(maxIter = 100, dualTolerance = 1e-9)
    val (_, value, _) = optimizer.maximize(f, BSV.fill(5)(0.1), 1)
    Assert.assertTrue(value.primalObjective == 0.0)
  }
}

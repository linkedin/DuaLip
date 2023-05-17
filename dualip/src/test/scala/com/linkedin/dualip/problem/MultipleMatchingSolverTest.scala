package com.linkedin.dualip.problem

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.maximizer.solver.firstorder.gradientbased.AcceleratedGradientDescent
import com.linkedin.dualip.projection.{BoxCutProjection, GreedyProjection, SimplexProjection}
import com.linkedin.dualip.slate.MultipleMatchingSlateComposer
import com.linkedin.spark.common.lib.TestUtils
import org.apache.spark.sql.SparkSession
import org.testng.Assert
import org.testng.annotations.Test

import scala.collection.mutable

// schema for DataFrame in primal test
case class MultipleMatchingSolverTestVar(item: Int, value: Double)

case class MultipleMatchingSolverTestRow(blockId: String, variables: Array[MultipleMatchingSolverTestVar])

class MultipleMatchingSolverTest {
  val enableHighDimOptimization = false // unit tests should work with both true/false settings, only using "false" to save runtime
  val matchingConstraintsPerIndex = 2
  val c: Map[Int, Double] = Map(
    0 -> -0.307766110869125, 1 -> -0.483770735096186, 2 -> -0.624996477039531, 3 -> -0.669021712383255, 4 -> -0.535811153938994,
    10 -> -0.257672501029447, 11 -> -0.812402617651969, 12 -> -0.882165518123657, 13 -> -0.204612161964178, 14 -> -0.710803845431656,
    20 -> -0.552322433330119, 21 -> -0.370320537127554, 22 -> -0.28035383997485, 23 -> -0.357524853432551, 24 -> -0.538348698290065,
    30 -> -0.0563831503968686, 31 -> -0.546558595029637, 32 -> -0.398487901547924, 33 -> -0.359475114848465, 34 -> -0.74897222686559,
    40 -> -0.468549283919856, 41 -> -0.170262051047757, 42 -> -0.76255108229816, 43 -> -0.690290528349578, 44 -> -0.420101450523362
  )
  val a: Map[(Int, Int, Int), Double] = Map(
    (0, 0, 0) -> 0.307766110869125, (0, 10, 0) -> 0.257672501029447, (0, 20, 0) -> 0.552322433330119, (0, 30, 0) -> 0.0563831503968686, (0, 40, 0) -> 0.468549283919856,
    (0, 0, 1) -> 0.307766110869125, (0, 10, 1) -> 0.257672501029447, (0, 20, 1) -> 0.552322433330119, (0, 30, 1) -> 0.0563831503968686, (0, 40, 1) -> 0.468549283919856,
    (1, 1, 0) -> 0.483770735096186, (1, 11, 0) -> 0.812402617651969, (1, 21, 0) -> 0.370320537127554, (1, 31, 0) -> 0.546558595029637, (1, 41, 0) -> 0.170262051047757,
    (1, 1, 1) -> 0.483770735096186, (1, 11, 1) -> 0.812402617651969, (1, 21, 1) -> 0.370320537127554, (1, 31, 1) -> 0.546558595029637, (1, 41, 1) -> 0.170262051047757,
    (2, 2, 0) -> 0.624996477039531, (2, 12, 0) -> 0.882165518123657, (2, 22, 0) -> 0.28035383997485, (2, 32, 0) -> 0.398487901547924, (2, 42, 0) -> 0.76255108229816,
    (2, 2, 1) -> 0.624996477039531, (2, 12, 1) -> 0.882165518123657, (2, 22, 1) -> 0.28035383997485, (2, 32, 1) -> 0.398487901547924, (2, 42, 1) -> 0.76255108229816,
    (3, 3, 0) -> 0.669021712383255, (3, 13, 0) -> 0.204612161964178, (3, 23, 0) -> 0.357524853432551, (3, 33, 0) -> 0.359475114848465, (3, 43, 0) -> 0.690290528349578,
    (3, 3, 1) -> 0.669021712383255, (3, 13, 1) -> 0.204612161964178, (3, 23, 1) -> 0.357524853432551, (3, 33, 1) -> 0.359475114848465, (3, 43, 1) -> 0.690290528349578,
    (4, 4, 0) -> 0.535811153938994, (4, 14, 0) -> 0.710803845431656, (4, 24, 0) -> 0.538348698290065, (4, 34, 0) -> 0.74897222686559, (4, 44, 0) -> 0.420101450523362,
    (4, 4, 1) -> 0.535811153938994, (4, 14, 1) -> 0.710803845431656, (4, 24, 1) -> 0.538348698290065, (4, 34, 1) -> 0.74897222686559, (4, 44, 1) -> 0.420101450523362
  )
  val metadata: Map[String, Double] = Map[String, Double]("boxCut" -> 2)
  val data: Seq[MultipleMatchingData] = (0 to 4).map(i => MultipleMatchingData(i.toString, (0 to 4).map(j => j + 10 * i)
    .map(j => (j % 10, c(j), Seq((0, a((j % 10, j, 0))), (1, a((j % 10, j, 1)))))), metadata))

  val indices: Array[Int] = (0 to 9).toArray
  val values: Array[Double] = Array.fill(10) {
    1.0
  }
  val b = new BSV(indices, values, values.length)
  val gamma = 1E-6

  // Expected values for this problem were computed with SCS
  val expectedDualObjective: Double = -3.5640
  val expectedLambda: Array[Double] = Array(0.0000000, 0.3327713, 0.3855439, 0.3212216, 0.5130992, 0.0000000, 0.3327713,
    0.3855439, 0.3212216, 0.5130992)
  val expectedPrimalUpperBound: Double = -(0.307766110869125 + 0.204612161964178 + 0.28035383997485 + 0.0563831503968686 + 0.170262051047757)

  @Test
  def testPrimal(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    import spark.implicits._
    spark.sparkContext.setLogLevel("warn")

    val multipleMatchingSlateOptimizer: MultipleMatchingSlateComposer = new MultipleMatchingSlateComposer(0, new GreedyProjection())
    val f = new MultipleMatchingSolverDualObjectiveFunction(spark.createDataset(data), b,
      matchingConstraintsPerIndex, multipleMatchingSlateOptimizer, gamma, enableHighDimOptimization, None)

    // compute value using calculate function
    val value = f.calculate(BSV(expectedLambda), mutable.Map.empty, 1)

    // alternative computation using returned primal solution and hardcoded objective
    val primal = f.getPrimalForSaving(BSV(expectedLambda))
      .get.as[MultipleMatchingSolverTestRow].collect.flatMap { r =>
      r.variables.map { x =>
        (r.blockId.toInt, x.item, x.value)
      }
    }
    val primalObj = primal.map { case (i, j, x) => x * c(i * 10 + j) + gamma * x * x / 2.0 }.sum
    Assert.assertTrue(Math.abs(value.primalObjective - primalObj) < 1E-8)
  }

  @Test
  def testSimplexSolver(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    import spark.implicits._
    spark.sparkContext.setLogLevel("warn")

    val multipleMatchingSlateOptimizer: MultipleMatchingSlateComposer = new MultipleMatchingSlateComposer(gamma,
      new SimplexProjection())
    val f = new MultipleMatchingSolverDualObjectiveFunction(spark.createDataset(data), b,
      matchingConstraintsPerIndex, multipleMatchingSlateOptimizer, gamma, enableHighDimOptimization, None)

    val optimizer = new AcceleratedGradientDescent(maxIter = 200)

    val (_, value, _) = optimizer.maximize(f, BSV.fill(10)(0.1))
    Assert.assertTrue(Math.abs(value.dualObjective - expectedDualObjective) < 0.01)
  }

  @Test
  def testBoxCutSolver(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    import spark.implicits._
    spark.sparkContext.setLogLevel("warn")

    val multipleMatchingSlateOptimizer: MultipleMatchingSlateComposer = new MultipleMatchingSlateComposer(gamma,
      new BoxCutProjection(1000))
    val f = new MultipleMatchingSolverDualObjectiveFunction(spark.createDataset(data), b, matchingConstraintsPerIndex,
      multipleMatchingSlateOptimizer, gamma, enableHighDimOptimization, None)

    val optimizer = new AcceleratedGradientDescent(maxIter = 200)

    val (_, value, _) = optimizer.maximize(f, BSV.fill(10)(0.1))
    Assert.assertTrue(Math.abs(value.dualObjective - (-5)) < 0.01)
  }
}
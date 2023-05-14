package com.linkedin.dualip.objective.distributedobjective

import com.linkedin.spark.common.lib.TestUtils
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.testng.Assert
import org.testng.annotations.Test

import scala.util.Random

class DistributedNonDifferentiableObjectiveTest {
  val lambdaDim = 10  // number of items
  val numSlots = 8    // number of slots for the user/request
  val partialSubgradientsTestData: Seq[SlateNonDifferentiable] = (0 until lambdaDim).map(itemId =>
    randomPartialSubgradient(itemId = itemId, numSlots = numSlots))
  val expectedCx: Double = partialSubgradientsTestData.map(_.objective).sum
  val expectedAx: Map[Int, Double] = partialSubgradientsTestData.map(x => (x.itemId, x.cost)).toMap

  @Test
  def testTwoStepSubGradientAggregation(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    val ds = spark.createDataset(partialSubgradientsTestData).repartition(2)
    val aggSubgradientStats = DistributedNonDifferentiableObjective.twoStepSubgradientAggregator(ds, lambdaDim, 2)
    assertAlmostEqual(aggSubgradientStats.dualSubgradient.toMap, expectedAx)
    assertAlmostEqual(aggSubgradientStats.objective, expectedCx)
  }

  @Test
  def testGuessNumberOfLambdaPartitions(): Unit = {
    val conf = new SparkConf()
    conf.set("spark.default.parallelism", "10")
    implicit val spark: SparkSession = TestUtils.createSparkSession(sparkConf = conf)

    Assert.assertEquals(guessNumberOfLambdaPartitions(1000), 40) // 4 * parallelism
    Assert.assertEquals(guessNumberOfLambdaPartitions(10), 10) // limited by lambda dimensionality
  }

  // Util methods for the test

  def randomPartialSubgradient(itemId: Int , numSlots: Int): SlateNonDifferentiable = {
    SlateNonDifferentiable(
      itemId = itemId,
      cost = Random.nextDouble(),
      objective = Random.nextDouble(),
      slots = (0 until numSlots).map(slotId => (slotId, Random.nextDouble()))
    )
  }

  def assertAlmostEqual(l: Double, r: Double): Unit = {
    Assert.assertTrue(math.abs(l - r) < 1e-10, s"$l != $r")
  }

  def assertAlmostEqual(l: Map[Int, Double], r: Map[Int, Double]): Unit = {
    Assert.assertEquals(l.size, r.size)
    l.keySet.foreach { key => assertAlmostEqual(l(key), r(key)) }
  }
}
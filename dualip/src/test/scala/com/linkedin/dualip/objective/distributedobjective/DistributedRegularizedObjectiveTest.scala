package com.linkedin.dualip.objective.distributedobjective

import com.linkedin.dualip.objective.PartialPrimalStats
import com.linkedin.spark.common.lib.TestUtils
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.testng.Assert
import org.testng.annotations.Test

import scala.util.Random

class DistributedRegularizedObjectiveTest {
  val partialGradientsTestData: Seq[PartialPrimalStats] = (0 until 10).map(_ => randomPartialGradient(dim = 9, nonZeroPositions = 8))
  val expectedCx: Double = partialGradientsTestData.map(_.objective).sum
  val expectedAx: Map[Int, Double] = partialGradientsTestData.flatMap(_.costs).groupBy(_._1).mapValues(_.map(_._2).sum)
  val expectedXx: Double = partialGradientsTestData.map(_.xx).sum

  @Test
  def testTwoStepGradientAggregation(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    val ds = spark.createDataset(partialGradientsTestData).repartition(2)
    val aggPrimalStats = DistributedRegularizedObjective.twoStepGradientAggregator(ds, 9, 2)
    assertAlmostEqual(aggPrimalStats.costs.toMap, expectedAx)
    assertAlmostEqual(aggPrimalStats.objective, expectedCx)
    assertAlmostEqual(aggPrimalStats.xx, expectedXx)
  }

  @Test
  def testGuessNumberOfLambdaPartitions(): Unit = {
    import DistributedRegularizedObjective.guessNumberOfLambdaPartitions
    val conf = new SparkConf()
    conf.set("spark.default.parallelism", "10")
    implicit val spark: SparkSession = TestUtils.createSparkSession(sparkConf = conf)

    Assert.assertEquals(guessNumberOfLambdaPartitions(1000), 40) // 4 * parallelism
    Assert.assertEquals(guessNumberOfLambdaPartitions(10), 10) // limited by lambda dimensionality
  }

  // Util methods for the test

  def randomPartialGradient(dim: Int, nonZeroPositions: Int): PartialPrimalStats = {
    PartialPrimalStats(
      costs = (0 until nonZeroPositions).map(_ => (Random.nextInt(dim), Random.nextInt(100) / 100.0)).toArray,
      objective = Random.nextDouble(),
      xx = Random.nextDouble()
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
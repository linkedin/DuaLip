package com.linkedin.dualip.objective.distributedobjective

import com.linkedin.dualip.objective.PartialPrimalStats
import com.linkedin.dualip.objective.distributedobjective.DistributedRegularizedObjective.accumulateSufficientStatistics
import com.linkedin.dualip.util.ArrayAggregation.partitionBounds
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
  def testAccumulateSufficientStatistics(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession("testAccumulateSufficientStatistics")
    import spark.implicits._

    val lambdaDim = 9
    val numPartitions = 4
    val primalStats = spark.createDataset(partialGradientsTestData)
    val aggregatedStats = accumulateSufficientStatistics(primalStats, lambdaDim, 4).toMap

    (0 until numPartitions - 1).foreach { partitionNumber =>
      val (startIndex, endIndex) = partitionBounds(arrayLength = lambdaDim + 2, numPartitions = numPartitions,
        partition = partitionNumber)
      (startIndex until endIndex).zipWithIndex.foreach { case (arrayIndex, index) =>
        Assert.assertEquals(aggregatedStats(partitionNumber)(index), expectedAx(arrayIndex), 0.01)
      }
    }
  }

  @Test
  def testTwoStepGradientAggregation(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession("testTwoStepGradientAggregation")
    import spark.implicits._

    Array(2, 5, 9).foreach { numPartitions =>
      print("number of partitions " + numPartitions + "\n")
      val ds = spark.createDataset(partialGradientsTestData).repartition(numPartitions)
      val aggPrimalStats = DistributedRegularizedObjective.twoStepGradientAggregator(ds, 9, numPartitions)
      assertAlmostEqual(aggPrimalStats.costs.toMap, expectedAx)
      assertAlmostEqual(aggPrimalStats.objective, expectedCx)
      assertAlmostEqual(aggPrimalStats.xx, expectedXx)
    }
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
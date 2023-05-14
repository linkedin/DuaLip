package com.linkedin.dualip.problem

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.spark.common.lib.TestUtils
import org.apache.spark.sql.SparkSession
import org.testng.Assert
import org.testng.annotations.Test

import scala.collection.mutable

case class MultiSlateMatchingPrimal(value: Double, items: Array[Int])

case class MultiSlateMatchingPrimalRow(blockId: String, variables: Array[MultiSlateMatchingPrimal])

class MultiSlateMatchingSolverTest {

  implicit val spark: SparkSession = TestUtils.createSparkSession()

  spark.sparkContext.setLogLevel("warn")

  // unit tests should work with both true/false settings, only using "false" to save runtime
  val enableHighDimOptimization = false

  val objTolerance = 1E-5
  val primalTolerance = 1E-8
  val dualTolerance = 1E-10
  val gamma = 0.01

  val numberOfBlocks = 3
  val numberOfItems = 3
  val slotsPerBlock: Seq[Int] = List(3, 2, 1)
  val metadata: Seq[Map[String, Double]] = List(
    Map[String, Double]("boxCut" -> 3),
    Map[String, Double]("boxCut" -> 2),
    Map[String, Double]("simplexInequality" -> 1)
  )

  // key: (i, j, k): cost for item j in slot k for user i
  val AMat: Map[(Int, Int, Int), Double] = Map(
    // budget constraint 0
    (0, 0, 0) -> 2, (0, 0, 1) -> 2, (0, 0, 2) -> 2,
    (1, 0, 0) -> 3, (1, 0, 1) -> 3,
    (2, 0, 0) -> 1,

    // budget constraint 1
    (0, 1, 0) -> 1, (0, 1, 1) -> 1, (0, 1, 2) -> 1,
    (1, 1, 0) -> 2, (1, 1, 1) -> 2,
    (2, 1, 0) -> 2,

    // budget constraint 2
    (0, 2, 0) -> 4, (0, 2, 1) -> 4, (0, 2, 2) -> 4,
    (1, 2, 0) -> 5, (1, 2, 1) -> 5,
    (2, 2, 0) -> 2
  )

  // key: (i, j, k): cost for item j in slot k for user i
  val costVector: Map[(Int, Int, Int), Double] = Map(
    (0, 0, 0) -> -3, (0, 0, 1) -> -3, (0, 0, 2) -> -3,
    (0, 1, 0) -> -2, (0, 1, 1) -> -2, (0, 1, 2) -> -2,
    (0, 2, 0) -> -1, (0, 2, 1) -> -1, (0, 2, 2) -> -1,
    (1, 0, 0) -> -2, (1, 0, 1) -> -2,
    (1, 1, 0) -> -4, (1, 1, 1) -> -4,
    (1, 2, 0) -> -6, (1, 2, 1) -> -6,
    (2, 0, 0) -> -5,
    (2, 1, 0) -> -8,
    (2, 2, 0) -> -4
  )

  val data: Seq[MultiSlateMatchingData] = (0 until numberOfBlocks)
    .map(blockId =>
      MultiSlateMatchingData(
        blockId.toString,
        (0 until numberOfItems).map(itemId => { (  // Seq[(j, Seq[(k, c_ijk, a_ijk)])]
          itemId,
          (0 until slotsPerBlock(blockId)).map(slotId => (  // Seq[(k, c_ijk, a_ijk)]
            slotId,
            costVector(blockId, itemId, slotId),
            AMat(blockId, itemId, slotId)
          ))
        )}),
        metadata = metadata(blockId)
      )
    )

  // budget values
  val budget: BSV[Double] = BSV(Array(3.0, 5.0, 10.0))

  // initial values of the duals
  val initialValueDuals: BSV[Double] = BSV.fill(numberOfItems)(0.10)

  // ground truth values (computed with CVXPY)
  val expectedDuals: BSV[Double] = BSV(Array(0, 0.6553266, 0))
  val expectedPrimals: Array[(Int, Int, Double)] = Array(
    (0, 0, 1.0),
    (0, 1, 1.0),
    (0, 2, 1.0),
    (1, 1, 1.0),
    (1, 2, 1.0),
    (2, 1, 1.0)
  )

  def checkPrimalSolution(primals: Array[(Int, Int, Double)],
                          truePrimals: Array[(Int, Int, Double)]): Unit = {
    Assert.assertEquals(primals.length, truePrimals.length)
    primals.indices.foreach { i =>
      Assert.assertEquals(primals(i)._1, truePrimals(i)._1)
      Assert.assertEquals(primals(i)._2, truePrimals(i)._2)
      Assert.assertTrue(Math.abs(primals(i)._3 - truePrimals(i)._3) < primalTolerance)
    }
  }

  @Test
  def testPrimalObjective(): Unit = {
    // expected values for this problem are computed with CVXPY


    val f = new MultiSlateMatchingSolverDualObjectiveFunction(spark.createDataset(data), budget, gamma,
      enableHighDimOptimization, None)

    // compute value using calculate function
    val value = f.calculate(expectedDuals, mutable.Map.empty, 1)

    // alternative computation using returned primal solution and hard-coded objective
    val primal = f.getPrimalForSaving(expectedDuals).get
      .as[MultiSlateMatchingPrimalRow].collect()
      .flatMap { r =>
        r.variables.map { variables =>
          (r.blockId.toInt, variables.items.min, variables.value)
        }
      }
    val primalObj = primal.map { case (blockId, itemId, x) =>
      0.5 * gamma * x * x + x * (0 until slotsPerBlock(blockId)).map(slotId =>
        costVector(blockId, itemId, slotId)).sum / slotsPerBlock(blockId)
    }.sum
    Assert.assertTrue(Math.abs(value.primalObjective - primalObj) < objTolerance)
  }

  @Test
  def testPrimalSolution(): Unit = {
    val f = new MultiSlateMatchingSolverDualObjectiveFunction(spark.createDataset(data), budget, gamma,
      enableHighDimOptimization, None)
    val optimizer = new AcceleratedGradientDescent(maxIter = 1000, dualTolerance = dualTolerance)
    val (_, value, _) = optimizer.maximize(f, initialValueDuals, 1)
    val duals = value.lambda

    // compute primal values
    val primals = f.getPrimalForSaving(duals).get
      .as[MultiSlateMatchingPrimalRow].collect()
      .flatMap { r =>
        r.variables.map { variables =>
          (r.blockId.toInt, variables.items.min, variables.value)
        }
      }
    checkPrimalSolution(primals, expectedPrimals)
  }
}

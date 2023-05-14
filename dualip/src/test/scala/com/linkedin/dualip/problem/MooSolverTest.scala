package com.linkedin.dualip.problem

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.data.MooData
import com.linkedin.dualip.maximizer.solver.secondorder.LBFGSB
import com.linkedin.dualip.util.{MapReduceArray, MapReduceDataset}
import com.linkedin.spark.common.lib.TestUtils
import org.apache.spark.sql.SparkSession
import org.testng.Assert
import org.testng.annotations.Test

class MooSolverTest {
  val a: Array[Array[Double]] = Array(
    Array(0.0169149451190606, 0.578941531083547, 0.737497533927672, 0.833785757608712, 0.91751586268656, 0.0),
    Array(0.0510786153143272, 0.578941531083547, 0.737497533927672, 0.833785757608712, 0.91751586268656, 0.0),
    Array(0.077301750308834, 0.578941531083547, 0.737497533927672, 0.833785757608712, 0.91751586268656, 0.0),
    Array(0.0923298332607374, 0.578941531083547, 0.737497533927672, 0.833785757608712, 0.91751586268656, 0.0),
    Array(0.111096161138266, 0.578941531083547, 0.737497533927672, 0.833785757608712, 0.91751586268656, 0.0),
    Array(0.140564785175957, 0.578941531083547, 0.737497533927672, 0.833785757608712, 0.91751586268656, 0.0),
    Array(0.145131220528856, 0.578941531083547, 0.737497533927672, 0.833785757608712, 0.91751586268656, 0.0),
    Array(0.163967578508891, 0.578941531083547, 0.737497533927672, 0.833785757608712, 0.91751586268656, 0.0),
    Array(0.165696729999036, 0.578941531083547, 0.737497533927672, 0.833785757608712, 0.91751586268656, 0.0),
    Array(0.243720785295591, 0.578941531083547, 0.737497533927672, 0.833785757608712, 0.91751586268656, 0.0))

  val c: Array[Double] = Array(-1.0, -1.0, -1.0, -1.0, -1.0, 0)
  val data: Seq[MooData] = (0 to 9).map(i => MooData(i, Array(a(i)), c, -1))
  val b: Array[Double] = Array(0.419003697729204)
  val infeasible_b: Array[Double] = Array(-1.0)

  // True values for this problem can be computed theoretically
  val expectedLambda = 7.114157
  val expectedDualObjective: Double = -5.5

  @Test
  def testSolver(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    spark.sparkContext.setLogLevel("warn")

    val f = new MooSolverDualObjectiveFunction(MapReduceDataset[MooData](spark.createDataset(data)), BSV(b), 1e-6, ProjectionType.Simplex)

    val optimizer = new LBFGSB()
    val (lambda, value, _) = optimizer.maximize(f, BSV.fill(1)(0.1))
    Assert.assertTrue(Math.abs(lambda(0) - expectedLambda) < 1e-5)
    Assert.assertTrue(Math.abs(value.dualObjective - expectedDualObjective) < 1e-5)
  }

  @Test
  def testParallelSolver(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    spark.sparkContext.setLogLevel("warn")

    val f = new MooSolverDualObjectiveFunction(MapReduceArray[MooData](data.toArray), BSV(b), 1e-6, ProjectionType.Simplex)

    val optimizer = new LBFGSB()
    val (lambda, value, _) = optimizer.maximize(f, BSV.fill(1)(0.1))
    Assert.assertTrue(Math.abs(lambda(0) - expectedLambda) < 1e-5)
    Assert.assertTrue(Math.abs(value.dualObjective - expectedDualObjective) < 1e-5)
  }

  @Test
  def testInfeasibleSolver(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    spark.sparkContext.setLogLevel("warn")

    val f = new MooSolverDualObjectiveFunction(MapReduceDataset[MooData](spark.createDataset(data)), BSV(infeasible_b), 1e-6, ProjectionType.Simplex)

    val optimizer = new LBFGSB()
    val (_, value, _) = optimizer.maximize(f, BSV.fill(1)(0.1))
    Assert.assertTrue(f.checkInfeasibility(value.lambda, value.dualObjective))
    Assert.assertTrue(value.dualObjective > f.getPrimalUpperBound)
  }

  @Test
  def testInfeasibleParallelSolver(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    spark.sparkContext.setLogLevel("warn")

    val f = new MooSolverDualObjectiveFunction(MapReduceArray[MooData](data.toArray), BSV(infeasible_b), 1e-6, ProjectionType.Simplex)

    val optimizer = new LBFGSB()
    val (_, value, _) = optimizer.maximize(f, BSV.fill(1)(0.1))
    Assert.assertTrue(f.checkInfeasibility(value.lambda, value.dualObjective))
    Assert.assertTrue(f.checkInfeasibility(value.lambda, value.dualObjective))
    Assert.assertTrue(value.dualObjective > f.getPrimalUpperBound)
  }
}

package com.linkedin.dualip.solver.secondorder

import breeze.linalg.{SparseVector => BSV}
import org.testng.Assert
import org.testng.annotations.Test

class LBFGSTest {

  @Test
  def testSimpleSolverObjectiveFunction(): Unit = {
    val x = BSV(Array(1.0, 1.0))
    val result = new SimpleObjective().calculate(x)
    Assert.assertEquals(result.dualObjective, -40.0)
    Assert.assertEquals(result.dualGradient, BSV(Array(4.0, -12.0)))
  }

  @Test
  def testRosenbrockSolverObjectiveFunction(): Unit = {
    val x = BSV(Array(1.0, 1.0))
    val result = new RosenbrockObjective().calculate(x)
    Assert.assertEquals(result.dualObjective, -0.0)
    Assert.assertEquals(result.dualGradient, BSV(Array(0.0, 0.0)))
  }

  @Test
  def testSimpleSolverConstrained(): Unit = {
    val solver = new LBFGS(alpha = 0.001, maxIter = 30)
    val (solution, _, _) = solver.maximize(new SimpleObjective(), BSV(Array(0.0, 0.0)))
    val Array(x, y) = solution.toArray
    Assert.assertTrue(Math.abs(x - 3.0) < 1e-2)
    Assert.assertTrue(Math.abs(y - 0.0) < 1e-2)
  }

  @Test
  def testSimpleSolverUnconstrained(): Unit = {
    val solver = new LBFGS(alpha = 10000, maxIter = 30)
    val (solution, _, _) = solver.maximize(new SimpleObjective(), BSV(Array(0.0, 0.0)))
    val Array(x, y) = solution.toArray
    Assert.assertTrue(Math.abs(x - 3.0) < 1e-2)
    Assert.assertTrue(Math.abs(y + 5.0) < 1e-2)
  }

  @Test
  def testRosenbrockSolverUnconstrained(): Unit = {
    val solver = new LBFGS(alpha = 10000, maxIter = 30)
    val (solution, _, _) = solver.maximize(new RosenbrockObjective(), BSV(Array(0.0, 0.0)))
    val Array(x, y) = solution.toArray
    Assert.assertTrue(Math.abs(x - 1.0) < 1e-2)
    Assert.assertTrue(Math.abs(y - 1.0) < 1e-2)
  }

  @Test
  def testRosenbrockSolverConstrained(): Unit = {
    val solver = new LBFGS(alpha = 0.0001, maxIter = 30)
    val (solution, _, _) = solver.maximize(new RosenbrockObjective(shift = -2.0), BSV(Array(0.0, 0.0)))
    val Array(x, y) = solution.toArray
    Assert.assertTrue(Math.abs(x - 0.0) < 1e-2)
    Assert.assertTrue(Math.abs(y - 2.0) < 1e-2)
  }
}


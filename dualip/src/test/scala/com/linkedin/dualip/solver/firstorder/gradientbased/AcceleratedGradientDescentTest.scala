package com.linkedin.dualip.solver.firstorder.gradientbased

import breeze.linalg.{SparseVector => BSV}
import org.testng.Assert
import org.testng.annotations.Test

class AcceleratedGradientDescentTest {

  @Test
  def testObjectiveFunction(): Unit = {
    val x = BSV(Array(1.0, 1.0))
    val result = new SimpleObjective().calculate(x)
    Assert.assertEquals(result.dualObjective, -40.0)
    Assert.assertEquals(result.dualGradient, BSV(Array(4.0, -12.0)))
  }

  @Test
  def testSolver(): Unit = {
    val solver = new AcceleratedGradientDescent(maxIter = 1000, dualTolerance = 1e-10)
    val (solution, _, _) = solver.maximize(new SimpleObjective(), BSV(Array(0.0, 0.0)))
    val Array(x, y) = solution.toArray
    Assert.assertTrue(Math.abs(x - 3.0) < 1e-3)
    Assert.assertEquals(y, 0.0)
  }
}

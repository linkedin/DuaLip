package com.linkedin.dualip.solver.firstorder.subgradientbased

import breeze.linalg.{SparseVector => BSV}
import org.testng.Assert
import org.testng.annotations.Test

class SubgradientDescentTest {

  @Test
  def testSimpleSolver(): Unit = {
    val solver = new SubgradientDescent(maxIter = 1000, dualTolerance = 1e-10)
    val (solution, _, _) = solver.maximize(new SimpleObjective(), BSV(Array(0.0, 0.0)))
    val Array(x, y) = solution.toArray
    Assert.assertTrue(Math.abs(x - 3.0) < 1e-2)
    Assert.assertEquals(y, 0.0)
  }

  @Test
  def testSimpleNonDifferentiableSolver(): Unit = {
    val solver = new SubgradientDescent(maxIter = 1000, dualTolerance = 1e-20)
    val (solution, result, _) = solver.maximize(new SimpleNonDifferentiableObjective(), BSV(Array(0.0, 0.0)))
    val Array(x, y) = solution.toArray
    Assert.assertTrue(Math.abs(x - 2.0) < 1e-2)
    Assert.assertTrue(Math.abs(y - 1.0) < 1e-2)
    Assert.assertTrue(Math.abs(result.dualObjective - 2.0) < 1e-2)
  }
}

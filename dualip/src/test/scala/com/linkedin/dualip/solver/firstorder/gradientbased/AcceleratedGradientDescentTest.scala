package com.linkedin.dualip.solver.firstorder.gradientbased

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.maximizer.solver.firstorder.gradientbased.AcceleratedGradientDescent
import com.linkedin.dualip.objective.{Quadratic1DObjective, SimpleObjective}
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

  @Test
  def testQuadratic1DFunction(): Unit = {
    // This test checks the functionality of the initialStepSize parameter.
    // For Quadratic1DObjective, we know that the initial gradient is 6.0. So after one step, the solution should
    // be at 6.0 * initialStepSize.
    val initialGradient = 6.0
    val defaultStepSize = 1E-5
    val solverDefault = new AcceleratedGradientDescent(maxIter = 1)
    val (solutionDefault, _, _) = solverDefault.maximize(new Quadratic1DObjective(), BSV(Array(0.0)))
    Assert.assertEquals(solutionDefault(0), initialGradient * defaultStepSize, "Test fails for default initialStepSize")

    val newStepSize = 0.1
    val solverNewStepSize = new AcceleratedGradientDescent(maxIter = 1, initialStepSize = newStepSize)
    val (solutionNewStepSize, _, _) = solverNewStepSize.maximize(new Quadratic1DObjective(), BSV(Array(0.0)))
    Assert.assertEquals(solutionNewStepSize(0), initialGradient * newStepSize, "Test fails for new initialStepSize")
  }
}

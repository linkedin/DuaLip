package com.linkedin.dualip.objective

import breeze.linalg.{SparseVector => BSV}
import org.testng.Assert
import org.testng.annotations.Test

import scala.collection.mutable

/**
  * Just a simple 2-d objective function f = -(x-3)^2 - (y+5)^2
  * because we maximize subject to x>=0 and y>=0
  * maximum is at x=3, y=0, dualObjective = -25, there is no primalObjective
  */
class SimpleObjective() extends DualPrimalObjective {
  override def dualDimensionality: Int = 2

  override def calculate(lambda: BSV[Double], log: mutable.Map[String, String]=null, verbosity: Int = 1, designInequality: Boolean = true, mixedDesignPivotNum: Int = 0): DualPrimalComputationResult = {
    val Array(x,y) = lambda.toArray
    val obj = -(x - 3.0)*(x - 3.0) - (y + 5.0)*(y + 5.0)
    val grad = Array(-2.0 * (x - 3.0), -2.0 * (y + 5.0))
    // primal, slack and maxSlack are dummy, they are used for logging and extra convergence criteria,
    // so they should not impact the testing of basic functionality
    DualPrimalComputationResult(lambda, obj, obj, BSV(grad), 0.0, BSV(Array(0.0, 0.0)), SlackMetadata(null, 0.0, 0.0, 0.0, 0.0))
  }

  @Test
  def testObjectiveFunction(): Unit = {
    val x = BSV(Array(1.0, 1.0))
    val result = new SimpleObjective().calculate(x)
    Assert.assertEquals(result.dualObjective, -40.0)
    Assert.assertEquals(result.dualGradient, BSV(Array(4.0, -12.0)))
  }
}

/**
  * https://en.wikipedia.org/wiki/Rosenbrock_function
  * The function is defined by
  * f(x,y)=(a-x)^{2}+b(y-x^{2})^{2}
  * It has a global minimum at (x,y)=(a,a^{2}), where f(x,y)=0.
  * Usually these parameters are set such that a=1, b=100
  */
class RosenbrockObjective(val shift: Double = 0.0) extends DualPrimalObjective {
  override def dualDimensionality: Int = 2

  override def calculate(lambda: BSV[Double], log: mutable.Map[String, String] = null, verbosity: Int = 1, designInequality: Boolean = true, mixedDesignPivotNum: Int = 0): DualPrimalComputationResult = {
    val Array(_x, _y) = lambda.toArray
    val x = _x - shift
    val y = _y - shift
    val obj = -(1 - x) * (1 - x) - 100 * (y - x * x) * (y - x * x)
    val grad = Array(-2.0 * ((1 - x) * -1) - 2.0 * 100 * (y - x * x) * (-2.0 * x), -2.0 * 100 * (y - x * x))
    // primal, slack and maxSlack are dummy, they are used for logging and extra convergence criteria,
    // so they should not impact the testing of basic functionality
    DualPrimalComputationResult(lambda, obj, obj, BSV(grad), 0.0, BSV(Array(0.0, 0.0)), SlackMetadata(null, 0.0, 0.0, 0.0, 0.0))
  }
}

/**
 * A simple function with non-differentiability at (0, 0) and (1, 0)
 * The function is defined by f(x, y) = g(x) - (y - 1)^2, where
 * g(x) = 2x if x <= 0,
 * = x  if 0 < x <= 2,
 * = -x + 4  if 2 < x.
 * Function maximized at x = 2, y = 1, max function value is 2.
 */
class SimpleNonDifferentiableObjective() extends DualPrimalObjective {
  override def dualDimensionality: Int = 2

  def getFunctionValue(x: Double, y: Double): Double = {
    if (x < 0) {
      2 * x - (y - 1) * (y - 1)
    } else if (x < 2) {
      x - (y - 1) * (y - 1)
    } else {
      -x + 4 - (y - 1) * (y - 1)
    }
  }

  def getGradient(x: Double, y: Double): Array[Double] = {
    val dy = -2 * (y - 1)
    val dx = x match {
      case x if x < 0 => 2
      case x if x == 0 => 1.5 // anything in [1, 2] works
      case x if x < 2 => 1
      case x if x == 2 => 0.5 // anything in [-1, 1] works
      case _ => -1
    }
    Array(dx, dy)
  }

  override def calculate(lambda: BSV[Double], log: mutable.Map[String, String] = null, verbosity: Int = 1, designInequality: Boolean = true, mixedDesignPivotNum: Int = 0): DualPrimalComputationResult = {
    val Array(x, y) = lambda.toArray
    val obj = getFunctionValue(x, y)
    val grad = getGradient(x, y)
    // primal, slack and maxSlack are dummy, they are used for logging and extra convergence criteria,
    // so they should not impact the testing of basic functionality
    DualPrimalComputationResult(lambda, obj, obj, BSV(grad), 0.0, BSV(Array(0.0, 0.0)), SlackMetadata(null, 0.0, 0.0, 0.0, 0.0))
  }
}
package com.linkedin.dualip.util

import com.linkedin.dualip.util.SolverUtility.{estimateLipschitzConstant, expandGroupedStepSize, stepSizeFromLipschitzConstants, updateDualGradientHistory}
import org.testng.Assert
import org.testng.annotations.Test

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

class SolverUtilityTest {
  /**
   * tests the utilities related to step-size calculation
   */
  val expectedStepSizes: Array[Double] = Array(0.1, 0.1, 0.1, 0.1, 1.0E-5, 1.0E-5)
  val expectedExpandedStepSizes: Array[Double] = Array(0.1, 0.1, 0.25, 0.25, 0.25, 0.35, 0.35, 0.35, 0.35, 0.35)
  val pivotPositionsForStepSize: Array[Int] = Array(2, 5)

  val gradient: Array[Double] = Array(0.5, 0.6)
  val lambda: Array[Double] = Array(0.8, 0.9)

  val expectedGradientHistory: ListBuffer[Array[Double]] = mutable.ListBuffer[Array[Double]]()
  expectedGradientHistory.append(Array(0.4, 0.5))
  expectedGradientHistory.append(Array(0.6, 0.7))
  expectedGradientHistory.append(Array(0.5, 0.6))

  val expectedLambdaHistory: ListBuffer[Array[Double]] = mutable.ListBuffer[Array[Double]]()
  expectedLambdaHistory.append(Array(0.35, 0.45))
  expectedLambdaHistory.append(Array(0.55, 0.65))
  expectedLambdaHistory.append(Array(0.8, 0.9))

  /**
   * checks whether the historical values for both gradients and lambdas are equal
   *
   * @param gradientHistory
   * @param lambdaHistory
   * @param maxHistoryLength
   */
  def checkEqualityOfGradientLambdaHistory(gradientHistory: ListBuffer[Array[Double]],
                                           lambdaHistory: ListBuffer[Array[Double]], maxHistoryLength: Int): Unit = {
    (0 until maxHistoryLength).foreach {
      index =>
        Assert.assertEquals(expectedGradientHistory(index), gradientHistory(index))
        Assert.assertEquals(expectedLambdaHistory(index), lambdaHistory(index))
    }
  }

  /**
   * checks the correctness of updateDualGradientHistory method when the length of history is less than the
   * maxHistoryLength
   */
  @Test
  def updateDualGradientHistoryTestOne(): Unit = {
    val gradientHistory: ListBuffer[Array[Double]] = mutable.ListBuffer[Array[Double]]()
    gradientHistory.append(Array(0.4, 0.5))
    gradientHistory.append(Array(0.6, 0.7))

    val lambdaHistory: ListBuffer[Array[Double]] = mutable.ListBuffer[Array[Double]]()
    lambdaHistory.append(Array(0.35, 0.45))
    lambdaHistory.append(Array(0.55, 0.65))

    updateDualGradientHistory(gradient, lambda, gradientHistory, lambdaHistory, 3)
    checkEqualityOfGradientLambdaHistory(gradientHistory, lambdaHistory, 3)
  }

  /**
   * checks the correctness of updateDualGradientHistory method when the length of history equals maxHistoryLength
   */
  @Test
  def updateDualGradientHistoryTestTwo(): Unit = {
    val gradientHistory: ListBuffer[Array[Double]] = mutable.ListBuffer[Array[Double]]()
    gradientHistory.append(Array(0.1, 0.2))
    gradientHistory.append(Array(0.4, 0.5))
    gradientHistory.append(Array(0.6, 0.7))

    val lambdaHistory: ListBuffer[Array[Double]] = mutable.ListBuffer[Array[Double]]()
    lambdaHistory.append(Array(0.15, 0.25))
    lambdaHistory.append(Array(0.35, 0.45))
    lambdaHistory.append(Array(0.55, 0.65))

    updateDualGradientHistory(gradient, lambda, gradientHistory, lambdaHistory, 3)
    checkEqualityOfGradientLambdaHistory(gradientHistory, lambdaHistory, 3)
  }

  @Test
  def estimateLipschitzConstantTest(): Unit = {
    val gradOne = Array(0.25, 0.50)
    val gradTwo = Array(0.10, 0.40)
    val dualOne = Array(0.1, 0.2)
    val dualTwo = Array(0.3, 0.4)
    val expectedLipschitzConstant = 0.6373
    Assert.assertEquals(expectedLipschitzConstant, estimateLipschitzConstant(gradOne, gradTwo, dualOne, dualTwo), 1e-4)
  }

  @Test
  def expandGroupedStepSizeTest(): Unit = {
    val groupedStepSize = Array(0.1, 0.25, 0.35)
    val expandedStepSizes = expandGroupedStepSize(pivotPositionsForStepSize, groupedStepSize, 10)
    Assert.assertEquals(expandedStepSizes, expectedExpandedStepSizes)
  }

  @Test
  def stepSizeFromLipschitzConstantsTest(): Unit = {
    val lipschitzConstantsOne = Seq(10.0, 20.0, 40.0)
    val expectedStepSizeOne = 0.025
    Assert.assertEquals(expectedStepSizeOne, stepSizeFromLipschitzConstants(lipschitzConstantsOne, 2, 1e-5, 0.1))

    val lipschitzConstantsTwo = Seq(1.0, 2.0, 4.0)
    val expectedStepSizeTwo = 0.1
    Assert.assertEquals(expectedStepSizeTwo, stepSizeFromLipschitzConstants(lipschitzConstantsTwo, 2, 1e-5, 0.1))

    val lipschitzConstantsThree = Seq(1.0, 2.0, 4.0)
    val expectedStepSizeThree = 1e-5
    Assert.assertEquals(expectedStepSizeThree, stepSizeFromLipschitzConstants(lipschitzConstantsThree, 10, 1e-5, 0.1))
  }

  def calculateGroupStepSize(gradient: Array[Double], lambda: Array[Double], gradientHistory: ListBuffer[Array[Double]], lambdaHistory: ListBuffer[Array[Double]], pivotPositionsForStepSize: Array[Int], i: Int, d: Double, d1: Double) = ???

  @Test
  def calculateStepSizeTest(): Unit = {
    val pivotPositionsForStepSize = Array(1, 3)
    val expectedStepSizes = Array(1.0, 0.6202, 0.2425)
    val gradient = Array(0.2, 0.3, 0.4, 0.5, 0.6)
    val lambda = Array(0.15, 0.25, 0.35, 0.45, 0.55)

    val gradientHistory: ListBuffer[Array[Double]] = mutable.ListBuffer[Array[Double]]()
    gradientHistory.append(Array(0.1, 0.2, 0.3, 0.4, 0.5))
    gradientHistory.append(Array(0.4, 0.5, 0.2, 0.1, 0.3))
    gradientHistory.append(Array(0.6, 0.7, 0.5, 0.4, 0.2))

    val lambdaHistory: ListBuffer[Array[Double]] = mutable.ListBuffer[Array[Double]]()
    lambdaHistory.append(Array(0.15, 0.25, 0.15, 0.45, 0.35))
    lambdaHistory.append(Array(0.35, 0.45, 0.15, 0.65, 0.45))
    lambdaHistory.append(Array(0.55, 0.65, 0.25, 0.35, 0.55))

    val stepSizes = calculateGroupStepSize(gradient, lambda, gradientHistory, lambdaHistory, pivotPositionsForStepSize,
      3, 1e-10, 1e10)
    (stepSizes zip expectedStepSizes).foreach { case (derivedStepSize, expectedStepSize) =>
      Assert.assertEquals(derivedStepSize, expectedStepSize, 1e-4)
    }
  }
}
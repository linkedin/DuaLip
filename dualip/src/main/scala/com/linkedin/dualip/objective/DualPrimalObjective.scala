package com.linkedin.dualip.objective

import breeze.linalg.{SparseVector => BSV}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable

/**
  * API for all optimizers.
  */
trait DualPrimalObjective {

  /**
    * The maximum value the primal formulation can take is pre-computed and stored.
    *
    * @return
    */
  def getPrimalUpperBound: Double = Double.PositiveInfinity

  /**
    *
    * @param lambda
    * @return
    */
  def getSardBound(lambda: BSV[Double]): Double = ???

  /**
    * Since the optimization is done in the dual space, if the dual objective ever exceeds the maximum primal value,
    * we know that the problem is infeasible and we can exit the optimizer.
    *
    * @param lambda
    * @param dualObjective
    * @return true if problem is infeasible, false otherwise.
    */
  def checkInfeasibility(lambda: BSV[Double], dualObjective: Double): Boolean = {
    val validLambda = lambda.forall(_ >= 0)
    validLambda && dualObjective > getPrimalUpperBound
  }

  /**
    * Dimensionality of the dual variable.
    *
    * @return
    */
  def dualDimensionality: Int

  /**
    * To add additional custom logging of Ax-b vector, one may want to log individual important constraints
    *
    * @param axMinusB - result of Ax-b computation
    * @param lambda   - current value of dual variable
    * @return - the map to be added to the iteration log (key is column name, value is column value)
    */
  def extraLogging(axMinusB: BSV[Double], lambda: BSV[Double]): Map[String, String] = {
    Map.empty
  }

  /**
    * Optional objective-specific logic to initialize lambda.
    *
    * @return
    */
  def getInitialLambda: Option[BSV[Double]] = None

  /**
    * Get primal for saving. The schema is problem specific, so we do not use the Dataset.
    * Not all solvers may support this functionality. Check the documentation of individual implementations.
    *
    * @param lambda The dual variable.
    * @return Optionally the DataFrame with primal solution. None if the functionality is not supported.
    */
  def getPrimalForSaving(lambda: BSV[Double]): Option[DataFrame] = {
    None
  }

  /**
    * Any custom finalization domain-specific logic can go here. For example, saving model in custom format.
    * Called by driver after the solver is done.
    *
    * @param lambda The dual variable.
    */
  def onComplete(lambda: BSV[Double]): Unit = {}

  /**
    * Calculate the dual objective, gradient and slack metadata using the current value of the parameters.
    *
    * @param lambda              The variable (vector) being optimized.
    * @param log                 Key-value pairs used to store logging information for each iteration of the optimizer.
    * @param verbosity           Control the logging level.
    * @param designInequality    True if Ax <= b (default), false if Ax = b or have mixed constraints.
    * @param mixedDesignPivotNum The pivot number if we have mixed A_1x <= b1 and A_2x = b2, i.e. how many inequality
    *                            constraints come first (default is 0).
    * @return Calculated values as class DualPrimalComputationResult.
    */
  def calculate(lambda: BSV[Double],
    log: mutable.Map[String, String],
    verbosity: Int,
    designInequality: Boolean = true,
    mixedDesignPivotNum: Int = 0
  ): DualPrimalComputationResult
}

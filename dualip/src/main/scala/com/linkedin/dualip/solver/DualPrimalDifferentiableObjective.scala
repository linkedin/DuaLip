package com.linkedin.dualip.solver

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.util.ProjectionType.ProjectionType
import com.linkedin.dualip.util.SolverUtility.SlackMetadata
import org.apache.spark.sql.{DataFrame, SparkSession}
import scala.collection.mutable

/**
  * Return type for gradient optimizer. It is more specific than the usual requirements for
  * gradient algorithms to support extra convergence criteria, based on primal properties.
  *
  * @param lambda             The dual variable: λ.
  * @param dualObjective      The dual objective with regularization: λ * (Ax -  b) + cx + x * x * gamma / 2.
  * @param dualObjectiveExact The dual objective without regularization: λ * (Ax -  b) + cx.
  * @param dualGradient       The gradient of the dual: (Ax -  b).
  * @param primalObjective    The primal objective: cx.
  * @param constraintsSlack   (Ax - b) value a byproduct of gradient computation, same shape as "b" and "dualGradient".
  * @param slackMetadata      Currently used for stopping criteria, logging, requires re-thinking.
  */
case class DualPrimalDifferentiableComputationResult(
  lambda: BSV[Double],
  dualObjective: Double,
  dualObjectiveExact: Double,
  dualGradient: BSV[Double],
  primalObjective: Double,
  constraintsSlack: BSV[Double],
  slackMetadata: SlackMetadata
)

/**
  * Return type for gradient optimizer. It is more specific than the usual requirements for
  * gradient algorithms to support extra convergence criteria, based on primal properties.
  *
  * The difference between this class and the previous class is that here we convert all the results in
  * SparseVector[Double] type into Array[(Int, Double)] to avoid any Encoder issue (SparseVector has no encoder
  * supported).
  *
  * @param lambda             The dual variable: λ.
  * @param dualObjective      The dual objective with regularization: λ * (Ax -  b) + cx + x * x * gamma / 2.
  * @param dualObjectiveExact The dual objective without regularization: λ * (Ax -  b) + cx.
  * @param dualGradient       The gradient of the dual: (Ax -  b).
  * @param primalObjective    The primal objective: cx.
  * @param constraintsSlack   (Ax - b) value a byproduct of gradient computation, same shape as "b" and "dualGradient".
  * @param slackMetadata      Currently used for stopping criteria, logging, requires re-thinking.
 */
case class DualPrimalDifferentiableComputationResultTuple(
  lambda: Array[(Int, Double)],
  dualObjective: Double,
  dualObjectiveExact: Double,
  dualGradient: Array[(Int, Double)],
  primalObjective: Double,
  constraintsSlack: Array[(Int, Double)],
  slackMetadata: SlackMetadata
)

/**
  * Return type which encapsulates DualPrimalDifferentiableComputationResultTuple as well as the log, dual, and
  * violation results in Lists.
  *
  * @param objectiveValue The DualPrimalDifferentiableComputationResultTuple value.
  * @param logList        The List which captures the log result.
  * @param dualList       The List which captures the dual result.
  * @param violationList  The List which captures the violation result.
  */
case class ResultWithLogsAndViolation(
  objectiveValue: DualPrimalDifferentiableComputationResultTuple,
  logList: List[String],
  dualList: List[(Int, Double)],
  violationList: List[(Int, Double)]
)

/**
  * API for all gradient optimizers.
  */
trait DualPrimalDifferentiableObjective {
  /**
    * Calculate the dual objective, gradient and slack metadata using the current value of the parameters.
    * @param lambda              The variable (vector) being optimized.
    * @param log                 Key-value pairs used to store logging information for each iteration of the optimizer.
    * @param verbosity           Control the logging level.
    * @param designInequality    True if Ax <= b (default), false if Ax = b or have mixed constraints.
    * @param mixedDesignPivotNum The pivot number if we have mixed A_1x <= b1 and A_2x = b2, i.e. how many inequality
    *                            constraints come first (default is 0).
    * @return Calculated values as class DualPrimalDifferentiableComputationResult.
    */
  def calculate(
    lambda: BSV[Double],
    log: mutable.Map[String, String],
    verbosity: Int,
    designInequality: Boolean = true,
    mixedDesignPivotNum: Int = 0
  ): DualPrimalDifferentiableComputationResult

  /**
    * The maximum value the primal formulation can take is pre-computed and stored.
    * @return
    */
  def getPrimalUpperBound: Double = Double.PositiveInfinity

  def getSardBound(lambda: BSV[Double]): Double = ???

  /**
    * Since the optimization is done in the dual space, if the dual objective ever exceeds the maximum primal value,
    * we know that the problem is infeasible and we can exit the optimizer.
    * @param value The result of evaluating the optimizer at the current value.
    * @return True if problem is infeasible, false otherwise.
    */
  def checkInfeasibility(value: DualPrimalDifferentiableComputationResult): Boolean = {
    val validLambda = value.lambda.forall(_ >= 0)
    validLambda && value.dualObjective > getPrimalUpperBound
  }

  /**
    * Dimensionality of the dual variable.
    * @return
    */
  def dualDimensionality: Int

  /**
    * Optional objective-specific logic to initialize lambda.
    * @return
    */
  def getInitialLambda: Option[BSV[Double]] = None

  /**
    * Get primal for saving. The schema is problem specific, so we do not use the Dataset.
    * Not all solvers may support this functionality. Check the documentation of individual implementations.
    * @param lambda The dual variable.
    * @return Optionally the DataFrame with primal solution. None if the functionality is not supported.
    */
  def getPrimalForSaving(lambda: BSV[Double]): Option[DataFrame] = {
    None
  }

  /**
   * Any custom finalization domain-specific logic can go here. For example, saving model in custom format.
   * Called by driver after the solver is done.
   * @param lambda The dual variable.
   */
  def onComplete(lambda: BSV[Double]): Unit = { }
}

/**
  * This is the loader API that will be implemented by companion objects.
  */
trait DualPrimalObjectiveLoader {
  /**
    * Very generic loader API, we don't know which arguments will be necessary for initialization
    * so we pass command line arguments and let the loader decide what it needs.
    * @param gamma          Currently used by all objectives, @todo think about making gamma a trait.
    * @param projectionType The type of projection used for simple constraints.
    * @param args           Custom args that are parsed by the loader.
    * @param spark          The spark session.
    * @return
    */
  def apply(gamma: Double, projectionType: ProjectionType, args: Array[String])
    (implicit spark: SparkSession): DualPrimalDifferentiableObjective
}
package com.linkedin.dualip.objective

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.util.SolverUtility.SlackMetadata

/**
  * Return type for gradient optimizer. It is more specific than the usual requirements for
  * gradient algorithms to support extra convergence criteria, based on primal properties.
  *
  * @param lambda                             The dual variable: λ.
  * @param dualObjective                      The dual objective:
  *                                           // for methods that use non-zero gamma, this is given by
  *                                           λ * (Ax -  b) + cx + x * x * gamma / 2.
  *                                           // for methods that do not use gamma, this is given by  λ * (Ax -  b) + cx
  *                                           and hence is same as the argument below
  * @param dualObjectiveWithoutRegularization The dual objective without regularization: λ * (Ax -  b) + cx.
  * @param dualGradient                       The gradient of the dual: (Ax -  b).
  * @param primalObjective                    The primal objective: cx.
  * @param constraintsSlack                   (Ax - b) value a byproduct of gradient computation, same shape as "b" and "dualGradient".
  * @param slackMetadata                      Currently used for stopping criteria, logging, requires re-thinking.
  */
case class DualPrimalComputationResult(
  lambda: BSV[Double],
  dualObjective: Double,
  dualObjectiveWithoutRegularization: Double,
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
  * @param lambda                             The dual variable: λ.
  * @param dualObjective                      The dual objective:
  *                                           // for methods that use non-zero gamma, this is given by
  *                                           λ * (Ax -  b) + cx + x * x * gamma / 2.
  *                                           // for methods that do not use gamma, this is given by  λ * (Ax -  b) + cx
  *                                           and hence is same as the argument below
  * @param dualObjectiveWithoutRegularization The dual objective without regularization: λ * (Ax -  b) + cx.
  * @param dualGradient                       The gradient of the dual: (Ax -  b).
  * @param primalObjective                    The primal objective: cx.
  * @param constraintsSlack                   (Ax - b) value a byproduct of gradient computation, same shape as "b" and "dualGradient".
  * @param slackMetadata                      Currently used for stopping criteria, logging, requires re-thinking.
  */
case class DualPrimalComputationResultTuple(
  lambda: Array[(Int, Double)],
  dualObjective: Double,
  dualObjectiveWithoutRegularization: Double,
  dualGradient: Array[(Int, Double)],
  primalObjective: Double,
  constraintsSlack: Array[(Int, Double)],
  slackMetadata: SlackMetadata
)

/**
  * Return type which encapsulates DualPrimalDifferentiableComputationResultTuple as well as the log, dual, and
  * violation results in Lists.
  *
  * @param objectiveValue The DualPrimalComputationResultTuple value.
  * @param logList        The List which captures the log result.
  * @param dualList       The List which captures the dual result.
  * @param violationList  The List which captures the violation result.
  */
case class ResultWithLogsAndViolation(
  objectiveValue: DualPrimalComputationResultTuple,
  logList: List[String],
  dualList: List[(Int, Double)],
  violationList: List[(Int, Double)]
)
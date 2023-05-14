package com.linkedin.dualip.objective

/**
  * Data encapsulates sufficient statistics of partial (distributed) primal solution,
  * necessary for gradient computation.
  * Note, the actual primal is not returned because it is not necessary for optimization
  * and its format may be problem-specific (e.g. slate optimization case).
  *
  * @param costs     - Ax vector (contribution of this segment into constraints), in sparse format
  * @param objective - contribution to objective cx (without the regularization term)
  * @param xx        - (x dot x) to compute solution norm
  */
case class PartialPrimalStats(costs: Array[(Int, Double)], objective: Double, xx: Double)

/**
  * Data encapsulates sufficient statistics of partial (distributed) primal solution, necessary for sub-gradient
  * computation.
  *
  * @param dualSubgradient - Ax vector (contribution of this segment into constraints), in sparse format
  * @param objective       - contribution to objective cx
  */
case class SubgradientStats(dualSubgradient: Array[(Int, Double)], objective: Double)

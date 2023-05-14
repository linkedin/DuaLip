package com.linkedin.dualip.maximizer

import com.linkedin.dualip.util.Status.Status

/**
  * Case class to represent optimizer state
  *
  * @param iterations - Number of iterations taken
  * @param status     - optimizer status, could be Running, Converged, Infeasible, Terminated or Failed
  * @param log        - log from optimizer
  */
case class OptimizerState(iterations: Int, status: Status, log: String)

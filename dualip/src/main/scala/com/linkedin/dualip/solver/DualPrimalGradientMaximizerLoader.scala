/*
 * BSD 2-CLAUSE LICENSE
 *
 * Copyright 2021 LinkedIn Corporation
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
 
package com.linkedin.dualip.solver

import com.linkedin.dualip.util.OptimizerType
import com.linkedin.dualip.util.OptimizerType.OptimizerType

/**
 * Initializes the gradient optimizer from input arguments.
 */
object DualPrimalGradientMaximizerLoader {
  def load(args: Array[String]): DualPrimalGradientMaximizer = {
    val params = DualPrimalGradientMaximizerParamsParser.parseArgs(args)
    import params._
    val solver: DualPrimalGradientMaximizer = solverType match {
      case OptimizerType.LBFGSB => new LBFGSB(maxIter = maxIter, dualTolerance = dualTolerance, slackTolerance = slackTolerance)
      case OptimizerType.LBFGS => new LBFGS(alpha = alpha, maxIter = maxIter, dualTolerance = dualTolerance, slackTolerance = slackTolerance)
      case OptimizerType.AGD => new AcceleratedGradientDescent(maxIter = maxIter, dualTolerance = dualTolerance, slackTolerance = slackTolerance)
    }
    solver
  }
}

/**
 * Union of optimizer parameters
 * @param solverType        Solver type
 * @param alpha             LBFGS positivity contstraint relaxation
 * @param dualTolerance     Tolerance criteria for dual variable change
 * @param slackTolerance    Tolerance criteria for slack
 * @param maxIter           Number of iterations
 */
case class DualPrimalGradientMaximizerParams(
  solverType: OptimizerType = OptimizerType.LBFGSB,
  alpha: Double = 1E-6,
  dualTolerance: Double = 1E-8,
  slackTolerance: Double = 5E-6,
  maxIter: Int = 100
)

/**
 * Parameters parser
 */
object DualPrimalGradientMaximizerParamsParser {
  def parseArgs(args: Array[String]): DualPrimalGradientMaximizerParams = {
    val parser = new scopt.OptionParser[DualPrimalGradientMaximizerParams]("Parsing optimizer params") {
      override def errorOnUnknownArgument = false
      val namespace = "optimizer"
      opt[String](s"$namespace.solverType") required() action { (x, c) => c.copy(solverType = OptimizerType.withName(x)) }
      opt[Double](s"$namespace.alpha") optional() action { (x, c) => c.copy(alpha = x) }
      opt[Double](s"$namespace.dualTolerance") required() action { (x, c) => c.copy(dualTolerance = x) }
      opt[Double](s"$namespace.slackTolerance") required() action { (x, c) => c.copy(slackTolerance = x) }
      opt[Int](s"$namespace.maxIter") required() action { (x, c) => c.copy(maxIter = x) }
    }

    parser.parse(args, DualPrimalGradientMaximizerParams()) match {
      case Some(params) => params
      case _ => throw new IllegalArgumentException(s"Parsing the command line arguments ${args.mkString(", ")} failed")
    }
  }
}
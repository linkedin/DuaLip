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
      case OptimizerType.AGD => new AcceleratedGradientDescent(maxIter = maxIter, dualTolerance = dualTolerance, slackTolerance = slackTolerance, designInequality = designInequality, mixedDesignPivotNum = mixedDesignPivotNum)
    }
    solver
  }
}

/**
 * Union of optimizer parameters
 * @param solverType           Solver type
 * @param designInequality     True if Ax <= b, false if Ax = b or have mixed constraints
 * @param mixedDesignPivotNum  The pivot number if we have mixed A_1x <= b1 and A_2x = b2, i.e. how many inequality constraints come first
 * @param alpha                LBFGS positivity constraint relaxation
 * @param dualTolerance        Tolerance criteria for dual variable change
 * @param slackTolerance       Tolerance criteria for slack
 * @param maxIter              Number of iterations
 */
case class DualPrimalGradientMaximizerParams(
  solverType: OptimizerType = OptimizerType.LBFGSB,
  designInequality: Boolean = true,
  mixedDesignPivotNum: Int = 0,
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
      opt[Boolean](s"$namespace.designInequality") optional() action { (x, c) => c.copy(designInequality = x) }
      opt[Int](s"$namespace.mixedDesignPivotNum") optional() action { (x, c) => c.copy(mixedDesignPivotNum = x) }
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
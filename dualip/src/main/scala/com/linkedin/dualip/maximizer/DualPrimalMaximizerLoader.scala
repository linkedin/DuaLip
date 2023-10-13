package com.linkedin.dualip.maximizer

import com.linkedin.dualip.maximizer.solver.firstorder.gradientbased.{AcceleratedGradientDescent, GradientDescent}
import com.linkedin.dualip.maximizer.solver.firstorder.subgradientbased.SubgradientDescent
import com.linkedin.dualip.maximizer.solver.secondorder.{LBFGS, LBFGSB}
import com.linkedin.dualip.util.OptimizerType
import com.linkedin.dualip.util.OptimizerType.OptimizerType
import scopt.OptionParser

/**
  * Initializes the gradient optimizer from input arguments.
  */
object DualPrimalMaximizerLoader {
  def load(args: Array[String]): DualPrimalMaximizer = {
    val params = DualPrimalMaximizerParamsParser.parseArgs(args)
    import params._
    val solver: DualPrimalMaximizer = solverType match {
      case OptimizerType.LBFGSB => new LBFGSB(maxIter = maxIter, dualTolerance = dualTolerance, slackTolerance = slackTolerance)
      case OptimizerType.LBFGS => new LBFGS(alpha = alpha, maxIter = maxIter, dualTolerance = dualTolerance, slackTolerance = slackTolerance)
      case OptimizerType.AGD => new AcceleratedGradientDescent(initialStepSize = initialStepSize, maxStepSize = maxStepSize,
        maxIter = maxIter, dualTolerance = dualTolerance,
        slackTolerance = slackTolerance, designInequality = designInequality, mixedDesignPivotNum = mixedDesignPivotNum,
        pivotPositionsForStepSize = pivotPositionsForStepSize)
      case OptimizerType.GD => new GradientDescent(initialStepSize = initialStepSize, maxStepSize = maxStepSize, maxIter = maxIter, dualTolerance = dualTolerance, slackTolerance = slackTolerance)
      case OptimizerType.SUBGD => new SubgradientDescent(initialStepSize = initialStepSize, maxStepSize = maxStepSize, maxIter = maxIter, dualTolerance = dualTolerance, slackTolerance = slackTolerance)
    }
    solver
  }
}

/**
  * Union of optimizer parameters
  *
  * @param solverType                Solver type
  * @param designInequality          True if Ax <= b, false if Ax = b or have mixed constraints
  * @param mixedDesignPivotNum       The pivot number if we have mixed A_1x <= b1 and A_2x = b2, i.e. how many inequality constraints come first
  * @param alpha                     LBFGS positivity constraint relaxation (optional)
  * @param initialStepSize           Initial step-size for gradient descent (optional)
  * @param maxStepSize               Maximum step-size for gradient descent (optional)
  * @param dualTolerance             Tolerance criteria for dual variable change
  * @param slackTolerance            Tolerance criteria for slack
  * @param maxIter                   Number of iterations
  * @param pivotPositionsForStepSize Pivot positions that mark different groups for which the step-sizes need to be tune
  */
case class DualPrimalMaximizerParams(solverType: OptimizerType = OptimizerType.LBFGSB,
  designInequality: Boolean = true,
  mixedDesignPivotNum: Int = 0,
  alpha: Double = 1E-6,
  initialStepSize: Double = 1E-5,
  maxStepSize: Double = 0.1,
  dualTolerance: Double = 1E-8,
  slackTolerance: Double = 5E-6,
  maxIter: Int = 100,
  pivotPositionsForStepSize: Array[Int] = Array(-1)
)

/**
  * Parameters parser
  */
object DualPrimalMaximizerParamsParser {
  def parseArgs(args: Array[String]): DualPrimalMaximizerParams = {
    val parser: OptionParser[DualPrimalMaximizerParams] = new
        scopt.OptionParser[DualPrimalMaximizerParams]("Parsing optimizer params") {
      override def errorOnUnknownArgument = false

      val namespace = "optimizer"
      opt[String](s"$namespace.solverType") required() action { (x, c) => c.copy(solverType = OptimizerType.withName(x)) }
      opt[Boolean](s"$namespace.designInequality") optional() action { (x, c) => c.copy(designInequality = x) }
      opt[Int](s"$namespace.mixedDesignPivotNum") optional() action { (x, c) => c.copy(mixedDesignPivotNum = x) }
      opt[Double](s"$namespace.alpha") optional() action { (x, c) => c.copy(alpha = x) }
      opt[Double](s"$namespace.initialStepSize") optional() action { (x, c) => c.copy(initialStepSize = x) }
      opt[Double](s"$namespace.maxStepSize") optional() action { (x, c) => c.copy(maxStepSize = x) }
      opt[Double](s"$namespace.dualTolerance") required() action { (x, c) => c.copy(dualTolerance = x) }
      opt[Double](s"$namespace.slackTolerance") required() action { (x, c) => c.copy(slackTolerance = x) }
      opt[Int](s"$namespace.maxIter") required() action { (x, c) => c.copy(maxIter = x) }
      opt[String](s"$namespace.pivotPositionsForStepSize") optional() action { (x, c) =>
        c.copy(pivotPositionsForStepSize = x.split(",").map(_.toInt))
      }
    }

    parser.parse(args, DualPrimalMaximizerParams()) match {
      case Some(params) => params
      case _ => throw new IllegalArgumentException(s"Parsing the command line arguments ${args.mkString(", ")} failed")
    }
  }
}
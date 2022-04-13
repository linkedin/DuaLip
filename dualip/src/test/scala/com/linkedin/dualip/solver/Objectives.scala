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

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.util.SolverUtility.SlackMetadata
import scala.collection.mutable

/**
  * Just a simple 2-d objective function f = -(x-3)^2 - (y+5)^2
  * because we maximize subject to x>=0 and y>=0
  * maximum is at x=3, y=0, dualObjective = -25, there is no primalObjective
  */
class SimpleObjective() extends DualPrimalDifferentiableObjective {
  override def dualDimensionality: Int = 2

  override def calculate(lambda: BSV[Double], log: mutable.Map[String, String]=null, verbosity: Int = 1, designInequality: Boolean = true, mixedDesignPivotNum: Int = 0): DualPrimalDifferentiableComputationResult = {
    val Array(x,y) = lambda.toArray
    val obj = -(x - 3.0)*(x - 3.0) - (y + 5.0)*(y + 5.0)
    val grad = Array(-2.0 * (x - 3.0), -2.0 * (y + 5.0))
    // primal, slack and maxSlack are dummy, they are used for logging and extra convergence criteria,
    // so they should not impact the testing of basic functionality
    DualPrimalDifferentiableComputationResult(lambda, obj, obj, BSV(grad), 0.0, BSV(Array(0.0, 0.0)), SlackMetadata(null, 0.0, 0.0, 0.0, 0.0))
  }
}

/**
  * https://en.wikipedia.org/wiki/Rosenbrock_function
  * The function is defined by
  * f(x,y)=(a-x)^{2}+b(y-x^{2})^{2}
  * It has a global minimum at (x,y)=(a,a^{2}), where f(x,y)=0.
  * Usually these parameters are set such that a=1, b=100
  */
class RosenbrockObjective(val shift: Double = 0.0) extends DualPrimalDifferentiableObjective {
  override def dualDimensionality: Int = 2

  override def calculate(lambda: BSV[Double], log: mutable.Map[String, String]=null, verbosity: Int = 1, designInequality: Boolean = true, mixedDesignPivotNum: Int = 0): DualPrimalDifferentiableComputationResult = {
    val Array(_x,_y) = lambda.toArray
    val x = _x - shift
    val y = _y - shift
    val obj = -(1 - x) * (1 - x) - 100 * (y - x * x) * (y - x * x)
    val grad = Array(-2.0 * ((1 - x) * -1) - 2.0 * 100 * (y - x * x) * (-2.0 * x), -2.0 * 100 * (y - x * x))
    // primal, slack and maxSlack are dummy, they are used for logging and extra convergence criteria,
    // so they should not impact the testing of basic functionality
    DualPrimalDifferentiableComputationResult(lambda, obj, obj, BSV(grad), 0.0, BSV(Array(0.0, 0.0)), SlackMetadata(null, 0.0, 0.0, 0.0, 0.0))
  }
}
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
 
package com.linkedin.dualip.problem

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.solver.AcceleratedGradientDescent
import com.linkedin.dualip.util.ProjectionType
import com.linkedin.spark.common.lib.TestUtils
import org.apache.spark.sql.SparkSession
import org.testng.Assert
import org.testng.annotations.Test

class MooSolverTest {
  val a = Array(
    Array(0.0169149451190606,  0.578941531083547,  0.737497533927672,  0.833785757608712,  0.91751586268656, 0.0),
    Array(0.0510786153143272,  0.578941531083547,  0.737497533927672,  0.833785757608712,  0.91751586268656, 0.0),
    Array(0.077301750308834,   0.578941531083547,  0.737497533927672,  0.833785757608712,  0.91751586268656, 0.0),
    Array(0.0923298332607374,  0.578941531083547,  0.737497533927672,  0.833785757608712,  0.91751586268656, 0.0),
    Array(0.111096161138266,   0.578941531083547,  0.737497533927672,  0.833785757608712,  0.91751586268656, 0.0),
    Array(0.140564785175957,   0.578941531083547,  0.737497533927672,  0.833785757608712,  0.91751586268656, 0.0),
    Array(0.145131220528856,   0.578941531083547,  0.737497533927672,  0.833785757608712,  0.91751586268656, 0.0),
    Array(0.163967578508891,   0.578941531083547,  0.737497533927672,  0.833785757608712,  0.91751586268656, 0.0),
    Array(0.165696729999036,   0.578941531083547,  0.737497533927672,  0.833785757608712,  0.91751586268656, 0.0),
    Array(0.243720785295591,   0.578941531083547,  0.737497533927672,  0.833785757608712,  0.91751586268656, 0.0))

  val c = Array(-1.0, -1.0, -1.0, -1.0, -1.0, 0)
  val data = (0 to 9).map(i => MooDataBlock(i, Array(a(i)), c))
  val b = Array(0.419003697729204)
  val infeasbile_b = Array(-1.0)

  // True values for this problem can be computed theoretically
  val expectedLambda = 7.114157
  val expectedDualObjective = -5.5

  @Test
  def testSolver(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    import spark.implicits._
    spark.sparkContext.setLogLevel("warn")

    val f = new MooSolverDualObjectiveFunction(spark.createDataset(data), BSV(b),1e-6, ProjectionType.Simplex)

    val optimizer = new AcceleratedGradientDescent()
    val (lambda, value, _) = optimizer.maximize(f, BSV.fill(1)(0.1))
    Assert.assertTrue(Math.abs(lambda(0) - expectedLambda) < 1e-3)
    Assert.assertTrue(Math.abs(value.dualObjective - expectedDualObjective) < 1e-3)
  }

  @Test
  def testInfeasibleSolver(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    import spark.implicits._
    spark.sparkContext.setLogLevel("warn")

    val f = new MooSolverDualObjectiveFunction(spark.createDataset(data), BSV(infeasbile_b),1e-6, ProjectionType.Simplex)

    val optimizer = new AcceleratedGradientDescent()
    val (_, value, _) = optimizer.maximize(f, BSV.fill(1)(0.1))
    Assert.assertTrue(f.checkInfeasibility(value))
    Assert.assertTrue(value.dualObjective > f.getPrimalUpperBound)
  }
}

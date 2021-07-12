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
import org.testng.Assert
import org.testng.annotations.Test

class LBFGSBTest {

  @Test
  def testSimpleSolverObjectiveFunction(): Unit = {
    val x = BSV(Array(1.0, 1.0))
    val result = new SimpleObjective().calculate(x)
    Assert.assertEquals(result.dualObjective, -40.0)
    Assert.assertEquals(result.dualGradient, BSV(Array(4.0, -12.0)))
  }

  @Test
  def testRosenbrockSolverObjectiveFunction(): Unit = {
    val x = BSV(Array(1.0, 1.0))
    val result = new RosenbrockObjective().calculate(x)
    Assert.assertEquals(result.dualObjective, -0.0)
    Assert.assertEquals(result.dualGradient, BSV(Array(0.0, 0.0)))
  }

  @Test
  def testSimpleSolverConstrained(): Unit = {
    val solver = new LBFGSB(maxIter = 30)
    val (solution, _, _) = solver.maximize(new SimpleObjective(), BSV(Array(0.0, 0.0)))
    val Array(x, y) = solution.toArray
    Assert.assertTrue(Math.abs(x - 3.0) < 1e-2)
    Assert.assertTrue(Math.abs(y - 0.0) < 1e-2)
  }

  @Test
  def testSimpleSolverUnconstrained(): Unit = {
    val solver = new LBFGSB(maxIter = 30)
    val (solution, _, _) = solver.maximize(new SimpleObjective(), BSV(Array(0.0, 0.0)))
    val Array(x, y) = solution.toArray
    // Unconstrained solution in this case is (3, -5) but it should not find it
    Assert.assertTrue(Math.abs(x - 3.0) < 1e-2)
    Assert.assertTrue(Math.abs(y - 0.0) < 1e-2)
  }

  @Test
  def testRosenbrockSolverUnconstrained(): Unit = {
    val solver = new LBFGSB(maxIter = 30)
    val (solution, _, _) = solver.maximize(new RosenbrockObjective(), BSV(Array(0.0, 0.0)))
    val Array(x, y) = solution.toArray
    Assert.assertTrue(Math.abs(x - 1.0) < 1e-2)
    Assert.assertTrue(Math.abs(y - 1.0) < 1e-2)
  }

  @Test
  def testRosenbrockSolverConstrained(): Unit = {
    val solver = new LBFGSB(maxIter = 30)
    val (solution, _, _) = solver.maximize(new RosenbrockObjective(shift = -2.0), BSV(Array(0.0, 0.0)))
    val Array(x, y) = solution.toArray
    Assert.assertTrue(Math.abs(x - 0.0) < 1e-2)
    Assert.assertTrue(Math.abs(y - 2.0) < 1e-2)
  }
}
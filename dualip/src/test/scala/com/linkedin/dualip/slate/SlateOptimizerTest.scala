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

package com.linkedin.dualip.slate

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.optimize.proximal.Constraint.PROBABILITYSIMPLEX
import breeze.optimize.proximal.QuadraticMinimizer
import com.linkedin.dualip.projection.{SimplexProjection, UnitBoxProjection}
import org.testng.annotations.Test
import org.testng.Assert

class SlateOptimizerTest {
  val db1 = DataBlock("0", Seq((0, -1.0, 0.0), (1, -1.0, 0.0)))
  val db2 = DataBlock("0", Seq((1, -1.0, 0.0), (0, -0.5, 0.0)))
  val db3 = DataBlock("0", Seq((0, 1.0, 0.0), (1, -0.5, 0.0), (2, -2.0, 0.0)))
  val db4 = DataBlock("0", Seq((0, 1.0, 0.0), (1, -1.1, 0.0)))
  val db5 = DataBlock("0", Seq((0, 1.0, 0.0), (1, 2.0, 0.0)))
  val db6 = DataBlock("0", Seq((0, -0.5, 0.0), (1, 0.1, 0.0)))
  val db7 = DataBlock("0", Seq((0, -0.5, 0.0), (1, -0.3, 0.0)))
  @Test
  def testSimplexOptimizer(): Unit = {
    val simplexProjection = new SimplexProjection()
    // Symmetric case
    val slates1 = new SingleSlotOptimizer(0.1, simplexProjection).optimize(db1, Array(0,0))
    Assert.assertEquals(slates1.size, 2)
    Assert.assertEquals(slates1(0).x, 0.5)
    Assert.assertEquals(slates1(1).x, 0.5)
    // Asymmetric case with small gamma (should pick largest element)
    val slates2 = new SingleSlotOptimizer(0.1, simplexProjection).optimize(db2, Array(0,0))
    Assert.assertEquals(slates2.size, 1)
    Assert.assertEquals(slates2(0).x, 1.0)
    Assert.assertEquals(slates2(0).costs, Seq((1, 0.0)))
    // Asymmetric case with large gamma (should pick both elements)
    val slates3 = new SingleSlotOptimizer(1, simplexProjection).optimize(db2, Array(0,0))
    Assert.assertEquals(slates3.size, 2)
    Assert.assertEquals(slates3(0).x, 0.75)
    Assert.assertEquals(slates3(1).x, 0.25)
  }

  @Test
  def testSimplexInequalityOptimizer(): Unit = {
    val simplexProjection = new SimplexProjection(inequality = true)
    // Symmetric case
    val slates1 = new SingleSlotOptimizer(0.1, simplexProjection).optimize(db1, Array(0,0))
    Assert.assertEquals(slates1.size, 2)
    Assert.assertEquals(slates1(0).x, 0.5)
    Assert.assertEquals(slates1(1).x, 0.5)
    // Asymmetric case with small gamma (should pick largest element)
    val slates2 = new SingleSlotOptimizer(0.1, simplexProjection).optimize(db2, Array(0,0))
    Assert.assertEquals(slates2.size, 1)
    Assert.assertEquals(slates2(0).x, 1.0)
    Assert.assertEquals(slates2(0).costs, Seq((1, 0.0)))
    // Asymmetric case with large gamma (should pick both elements)
    val slates3 = new SingleSlotOptimizer(1, simplexProjection).optimize(db2, Array(0,0))
    Assert.assertEquals(slates3.size, 2)
    Assert.assertEquals(slates3(0).x, 0.75)
    Assert.assertEquals(slates3(1).x, 0.25)

    val slates4 = new SingleSlotOptimizer(1, simplexProjection).optimize(db3, Array(0,0,0))
    Assert.assertEquals(slates4.size, 1)
    Assert.assertEquals(slates4(0).x, 1.0)
    Assert.assertEquals(slates4(0).costs, Seq((2, 0.0)))

    val slates5 = new SingleSlotOptimizer(1, simplexProjection).optimize(db4, Array(0,0))
    Assert.assertEquals(slates5.size, 1)
    Assert.assertEquals(slates5(0).x, 1.0)
    Assert.assertEquals(slates5(0).costs, Seq((1, 0.0)))

    val slates6 = new SingleSlotOptimizer(1, simplexProjection).optimize(db5, Array(0,0))
    Assert.assertEquals(slates6.size, 0)

    val slates7 = new SingleSlotOptimizer(1, simplexProjection).optimize(db6, Array(0,0))
    Assert.assertEquals(slates7.size, 1)
    Assert.assertEquals(slates7(0).x, 0.5)
    Assert.assertEquals(slates7(0).costs, Seq((0, 0.0)))

    val slates8 = new SingleSlotOptimizer(1, simplexProjection).optimize(db7, Array(0,0))
    Assert.assertEquals(slates8.size, 2)
    Assert.assertEquals(slates8(0).x, 0.5)
    Assert.assertEquals(slates8(1).x, 0.3)
  }

  @Test
  def testUnitBoxOptimizer(): Unit = {
    val unitBoxProjection = new UnitBoxProjection()
    // Both cases the negative element should be dropped.
    // Small gamma - scales the middle value towards the max
    val slates1 = new SingleSlotOptimizer(gamma=0.1, unitBoxProjection).optimize(db3, Array(0,0,0))
    Assert.assertEquals(slates1.size, 2)
    Assert.assertEquals(slates1(0).x, 1.0)
    Assert.assertEquals(slates1(1).x, 1.0)

    // Large gamma - keeps the middle value as is.
    val slates2 = new SingleSlotOptimizer(gamma=1, unitBoxProjection).optimize(db3, Array(0,0,0))
    Assert.assertEquals(slates2.size, 2)
    Assert.assertEquals(slates2(0).x, 0.5)
    Assert.assertEquals(slates2(1).x, 1.0)
  }
}

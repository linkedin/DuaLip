package com.linkedin.dualip.slate

import com.linkedin.dualip.projection.{QPProjection, SimplexProjection, UnitBoxProjection}
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
  def testQPOptimizer(): Unit = {
    val qpProjection = new QPProjection()
    // Symmetric case
    val slates1 = new SingleSlotOptimizer(0.1, qpProjection).optimize(db1, Array(0,0))
    Assert.assertEquals(slates1.size, 2)
    Assert.assertEquals(slates1(0).x, 0.5)
    Assert.assertEquals(slates1(1).x, 0.5)
    // Asymmetric case with small gamma (should pick largest element)
    val slates2 = new SingleSlotOptimizer(0.1, qpProjection).optimize(db2, Array(0,0))
    Assert.assertEquals(slates2.size, 1)
    Assert.assertEquals(slates2(0).x, 1.0)
    Assert.assertEquals(slates2(0).costs, Seq((1, 0.0)))
    // Asymmetric case with large gamma (should pick both elements)
    val slates3 = new SingleSlotOptimizer(1, qpProjection).optimize(db2, Array(0,0))
    Assert.assertEquals(slates3.size, 2)
    Assert.assertTrue(Math.abs(slates3(0).x - 0.75) < 1E-2)
    Assert.assertTrue(Math.abs(slates3(1).x - 0.25) < 1E-2)
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

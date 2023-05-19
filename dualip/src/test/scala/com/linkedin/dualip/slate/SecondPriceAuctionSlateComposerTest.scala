package com.linkedin.dualip.slate

import org.testng.Assert
import org.testng.annotations.Test

class SecondPriceAuctionSlateComposerTest {
  @Test
  def testSecondPriceDP(): Unit = {
    val testData = Seq(
      // Simple test, no dual costs
      (
        Array[Double](4, 3, 2, 1), // input costs
        Array[Double](0, 0, 0, 0), // input dual variables
        4, // slate size
        6.0, // expected objective
        Seq(0, 1, 2, 3), // expected ids list
        Seq(3, 2, 1, 0) // expected costs
      ),
      // High dual cost
      (
        Array[Double](10, 9, 8, 7, 6), // input costs
        Array[Double](0, 0.9, 0, 0, 0), // input dual variables
        3, // slate size
        15.0, // expected objective
        Seq(0, 2, 3), // expected ids list - skip second element because of high dual penalty.
        Seq(8, 7, 0) // expected costs
      )
    )

    testData.foreach { case (c, lambda, size, expectedObjective, expectedIds, expectedCosts) =>
      val (ids, costs, obj) = SecondPriceAuctionSlateComposer.secondPriceSolver(c, lambda, size)
      Assert.assertEquals(ids.toList, expectedIds)
      Assert.assertEquals(costs.toList, expectedCosts)
      Assert.assertEquals(obj, expectedObjective)
    }
  }
}

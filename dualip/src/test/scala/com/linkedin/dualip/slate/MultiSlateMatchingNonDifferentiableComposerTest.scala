package com.linkedin.dualip.slate

import org.testng.Assert
import org.testng.annotations.Test

import scala.math.log

class MultiSlateMatchingNonDifferentiableComposerTest {
  // Helper function: Compare one SlateNonDifferentiable object against another.
  def compareSingleSlate(expectedSlate: SlateNonDifferentiable, actualSlate: SlateNonDifferentiable): Boolean = {
    if ((expectedSlate.itemId != actualSlate.itemId) |
        (expectedSlate.cost != actualSlate.cost) |
        (expectedSlate.objective != actualSlate.objective))
      return false
    else {
      val expectedSlots = expectedSlate.slots
      val actualSlots = actualSlate.slots
      if (expectedSlots.length != actualSlots.length)
        return false
      else {
        (0 until expectedSlots.length).foreach { slotId =>
          val expectedTuple = expectedSlots(slotId)
          val actualTuple = actualSlots(slotId)
          if (expectedTuple._1 != actualTuple._1 | expectedTuple._2 != actualTuple._2)
            return false
        }
      }
    }
    true
  }

  // Helper function: Compare a list of SlateNonDifferentiable objects against another.
  def compareSlates(expectedSlates: Seq[SlateNonDifferentiable], actualSlates: Seq[SlateNonDifferentiable]): Boolean = {
    if (expectedSlates.length != actualSlates.length)
      return false
    else {
      (0 until expectedSlates.length).foreach { itemId =>
        if (!compareSingleSlate(expectedSlates(itemId), actualSlates(itemId))) {
          return false
        }
      }
    }
    true
  }

  /**
   * Test data for one user.
   * Assume that there are 5 campaigns in total, but campaign 2 is not involved in this particular request.
   */
  val itemIds: Array[Int] = Array(0, 1, 4, 3)

  // key: j: coefficient in objective for item j in first slot
  val cVector: Map[Int, Double] = Map(
    0 -> -4.0, 1 -> -2.0, 4 -> -5.0, 3 -> -1.0
  )

  // key: j: cost for item j in first slot (constraint)
  val AMat: Map[Int, Double] = Map(
    0 -> 1.0, 1 -> 2.0, 4 -> 3.0, 3 -> 4.0
  )

  // position effect factors (assuming first slot is slot 0)
  val e1 = 1.0 / (1.0 + log(1.0 + 1))
  val e2 = 1.0 / (1.0 + log(1.0 + 2))

  val singleSlotData: Array[(Int, Seq[(Int, Double, Double)])] = itemIds.map(
    itemId => (
      itemId, Seq((0, cVector(itemId), AMat(itemId)))
    )
  )

  val multiSlotData: Array[(Int, Seq[(Int, Double, Double)])] = itemIds.map(
    itemId => (
      itemId, Seq(
        (0, cVector(itemId), AMat(itemId)),
        (1, cVector(itemId) * e1, AMat(itemId) * e1),
        (2, cVector(itemId) * e2, AMat(itemId) * e2)
      )
    )
  )

  val multiSlateMatchingNonDifferentiableComposer = new MultiSlateMatchingNonDifferentiableComposer()

  @Test
  // Single slot test for duals = 0. Solution is 1 for item 4, 0 for the rest.
  def itemsSingleSlotArgminTestDualZero(): Unit = {
    val actualSlates: Seq[SlateNonDifferentiable] = multiSlateMatchingNonDifferentiableComposer.
      itemsSingleSlotArgmin(singleSlotData, duals = Array(0.0, 0.0, 0.0, 0.0, 0.0))
    val expectedSlates: Seq[SlateNonDifferentiable] = List(
      SlateNonDifferentiable(0, 0.0,  0.0, List((0, 0.0))),
      SlateNonDifferentiable(1, 0.0,  0.0, List((0, 0.0))),
      SlateNonDifferentiable(4, 3.0, -5.0, List((0, 1.0))),
      SlateNonDifferentiable(3, 0.0,  0.0, List((0, 0.0)))
    )
    Assert.assertTrue(compareSlates(expectedSlates, actualSlates))
  }

  @Test
  // Single slot test for non-zero duals. Solution is 1 for item 0, 0 for the rest.
  def itemsSingleSlotArgminTestNegative(): Unit = {
    val actualSlates: Seq[SlateNonDifferentiable] = multiSlateMatchingNonDifferentiableComposer.
      itemsSingleSlotArgmin(singleSlotData, duals = Array(2.0, 1.0, 0.0, 1.0, 2.0))
    val expectedSlates: Seq[SlateNonDifferentiable] = List(
      SlateNonDifferentiable(0, 1.0, -4.0, List((0, 1.0))),
      SlateNonDifferentiable(1, 0.0,  0.0, List((0, 0.0))),
      SlateNonDifferentiable(4, 0.0,  0.0, List((0, 0.0))),
      SlateNonDifferentiable(3, 0.0,  0.0, List((0, 0.0)))
    )
    Assert.assertTrue(compareSlates(expectedSlates, actualSlates))
  }

  // Single slot test for non-zero duals. Solution is 0 for all campaigns.
  @Test
  def itemsSingleSlotArgminTestZero(): Unit = {
    val actualSlates: Seq[SlateNonDifferentiable] = multiSlateMatchingNonDifferentiableComposer.
      itemsSingleSlotArgmin(singleSlotData, duals = Array(4.0, 1.0, 0.0, 1.0, 2.0))
    val expectedSlates: Seq[SlateNonDifferentiable] = List(
      SlateNonDifferentiable(0, 0.0, 0.0, List((0, 0.0))),
      SlateNonDifferentiable(1, 0.0, 0.0, List((0, 0.0))),
      SlateNonDifferentiable(4, 0.0, 0.0, List((0, 0.0))),
      SlateNonDifferentiable(3, 0.0, 0.0, List((0, 0.0)))
    )
    Assert.assertTrue(compareSlates(expectedSlates, actualSlates))
  }

  // Multi-slot test for duals = 0. Solution is item 4 in slot 1, item 0 in slot 2, item 1 in slot 3.
  @Test
  def itemsMultiSlotArgMaxTestDualZero(): Unit = {
    val actualSlates: Seq[SlateNonDifferentiable] = multiSlateMatchingNonDifferentiableComposer.
      itemsMultiSlotArgmin(multiSlotData, duals = Array(0.0, 0.0, 0.0, 0.0, 0.0), numSlots = 3)
    val expectedSlates: Seq[SlateNonDifferentiable] = List(
      SlateNonDifferentiable(4,      3.0,      -5.0, List((0, 1.0), (1, 0.0), (2, 0.0))),
      SlateNonDifferentiable(0, 1.0 * e1, -4.0 * e1, List((0, 0.0), (1, 1.0), (2, 0.0))),
      SlateNonDifferentiable(1, 2.0 * e2, -2.0 * e2, List((0, 0.0), (1, 0.0), (2, 1.0))),
      SlateNonDifferentiable(3,      0.0,       0.0, List((0, 0.0), (1, 0.0), (2, 0.0)))
    )
    Assert.assertTrue(compareSlates(expectedSlates, actualSlates))
  }

  // Multi-slot test for duals != 0. Solution is item 4 in slot 1, item 0 in slot 2, item 1 in slot 3.
  @Test
  def itemsMultiSlotArgMaxTestDualNonZero(): Unit = {
    val actualSlates: Seq[SlateNonDifferentiable] = multiSlateMatchingNonDifferentiableComposer.
      itemsMultiSlotArgmin(multiSlotData, duals = Array(2.0, 1.0, 0.0, 1.0, 2.0), numSlots = 3)
    val expectedSlates: Seq[SlateNonDifferentiable] = List(
      SlateNonDifferentiable(0,      1.0,      -4.0, List((0, 1.0), (1, 0.0), (2, 0.0))),
      SlateNonDifferentiable(1, 2.0 * e1, -2.0 * e1, List((0, 0.0), (1, 1.0), (2, 0.0))),
      SlateNonDifferentiable(4, 3.0 * e2, -5.0 * e2, List((0, 0.0), (1, 0.0), (2, 1.0))),
      SlateNonDifferentiable(3,      0.0,       0.0, List((0, 0.0), (1, 0.0), (2, 0.0)))
    )
    Assert.assertTrue(compareSlates(expectedSlates, actualSlates))
  }

  // Same data as in itemsSingleSlotArgminTestNegative().
  @Test
  def getSlateSingleSlot(): Unit = {
    val singleSlotDataBlock = MultiSlateMatchingData(id = "23", data = singleSlotData)
    val actualSlates = multiSlateMatchingNonDifferentiableComposer.
      getSlate(singleSlotDataBlock, duals = Array(2.0, 1.0, 0.0, 1.0, 2.0))
    val expectedSlates: Seq[SlateNonDifferentiable] = List(
      SlateNonDifferentiable(0, 1.0, -4.0, List((0, 1.0))),
      SlateNonDifferentiable(1, 0.0, 0.0, List((0, 0.0))),
      SlateNonDifferentiable(4, 0.0, 0.0, List((0, 0.0))),
      SlateNonDifferentiable(3, 0.0, 0.0, List((0, 0.0)))
    )
    Assert.assertTrue(compareSlates(expectedSlates, actualSlates))
  }

  // Same data as in itemsMultiSlotArgMaxTestDualNonZero().
  @Test
  def getSlateMultiSlot(): Unit = {
    val multiSlotDataBlock = MultiSlateMatchingData(id = "24", data = multiSlotData)
    val actualSlates = multiSlateMatchingNonDifferentiableComposer.
      getSlate(multiSlotDataBlock, duals = Array(2.0, 1.0, 0.0, 1.0, 2.0))
    val expectedSlates: Seq[SlateNonDifferentiable] = List(
      SlateNonDifferentiable(0, 1.0, -4.0, List((0, 1.0), (1, 0.0), (2, 0.0))),
      SlateNonDifferentiable(1, 2.0 * e1, -2.0 * e1, List((0, 0.0), (1, 1.0), (2, 0.0))),
      SlateNonDifferentiable(4, 3.0 * e2, -5.0 * e2, List((0, 0.0), (1, 0.0), (2, 1.0))),
      SlateNonDifferentiable(3, 0.0, 0.0, List((0, 0.0), (1, 0.0), (2, 0.0)))
    )
    Assert.assertTrue(compareSlates(expectedSlates, actualSlates))
  }
}

package com.linkedin.dualip.data

import com.linkedin.dualip.projection.Projection

/**
  * case class for the Slate variables used in the non-differentiable solver
  *
  * @param itemId    Item ID.
  * @param cost      Contribution to the gradient by the given item (sum over slots of a_ijk x_ijk).
  * @param objective Contribution to the objective by the given item (sum over slots of c_ijk x_ijk).
  * @param slots     Seq of tuples consisting of slot index and the primal values for a given item and slot, i.e.
  *                  (k, x_ijk).
  */
case class SlateNonDifferentiable(itemId: Int, cost: Double, objective: Double, slots: Seq[(Int, Double)])

/**
  * case class for the A-c data block corresponding to the multi-slate matching problem
  *
  * @param id       Block ID.
  * @param data     c-A data corresponding to a given block-id. The first entry of type Int in the data block is the
  *                 item ID. The second entry is a Seq of (Int, Double, Double) triplets, each corresponding to the
  *                 slot ID and the c and A values associated with that block ID, item ID and slot ID.
  * @param metadata : Projection metadata.
  *
  */
case class MultiSlateMatchingData(id: String, data: Seq[(Int, Seq[(Int, Double, Double)])],
  metadata: Projection#Metadata = null)

object MultiSlateMatchingData {
  val optionalFields: Seq[String] = Seq("metadata")
}
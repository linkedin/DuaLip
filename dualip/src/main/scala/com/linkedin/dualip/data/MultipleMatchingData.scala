package com.linkedin.dualip.data

import com.linkedin.dualip.projection.Projection

/**
  * case class for the A-c data block corresponding to the multiple matching problem
  *
  * @param id       : block-id
  * @param data     : c-A data corresponding to a given block-id. The first entry of type Int in the data block is the
  *                 item ID. The second entry of type Double is the c value associated with that (block, item). The
  *                 third entry is a Seq of (Int, Double) pairs, each corresponding to the constraint index and A value
  *                 for that (block, item, constraint).
  * @param metadata : projection metadata
  */
case class MultipleMatchingData(id: String, data: Seq[(Int, Double, Seq[(Int, Double)])],
  metadata: Projection#Metadata = null)

object MultipleMatchingData {
  val optionalFields: Seq[String] = Seq("metadata")
}
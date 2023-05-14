package com.linkedin.dualip.data

import com.linkedin.dualip.projection.Projection

/**
  * Representation of the data block, used in slate generation
  * The assumption is that spark.Dataset[MatchingData] is going to be used to store the input data,
  * optimized for fast algorithm iterations.
  *
  * @param id       - id of the block (i.e. impression id).
  * @param data     - sparse vector of tuples: (rowId, c(rowId), a(rowId))
  *                 c(rowId) - is the objective function component of the corresponding variable
  *                 a(rowId) - is the element of the constraint diagonal element
  * @param metadata - features or metadata to be used for each block (string to number) mapping.
  */
case class MatchingData(id: String, data: Seq[(Int, Double, Double)], metadata: Projection#Metadata = null)

object MatchingData {
  val optionalFields: Seq[String] = Seq("metadata")
}
package com.linkedin.dualip.util

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.projection.Projection

/**
  * case class for the A-G-c data block corresponding to the constrained matching problem
  *
  * @param id       : row-id
  * @param data     : c-A-G data corresponding to a given row-id
  * @param metadata : projection metadata
  *
  * The data field has the following structure.
  *
  * {
  "name": "data",
  "type": [
    {
      "type": "array",
      "items": [
        {
          "type": "record",
          "name": "data",
          "fields": [
            {
              "name": "colId",
              "type": "int"
            },
            {
              "name": "c",
              "type": "double"
            },
            {
              "name": "A",
              "type": "double"
            },
            {
              "name": "G",
              "type": "array",
              "items": [
                {
                  "fields": [
                    {
                      "name": "row-Id of G",
                      "type": "int"
                    },
                    {
                      "name": "value from G matrix",
                      "type": "double"
                    }
                  ]
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
  */
case class ConstrainedMatchingDataBlock(id: String, data: Seq[(Int, Double, Double, Seq[(Int, Double)])],
  metadata: Projection#Metadata = null)

object ConstrainedMatchingDataBlock {
  val optionalFields = Seq("metadata")
}

/**
  * case class for the budgets
  *
  * @param budgetLocal  : budget corresponding to local constraints
  * @param budgetGlobal : budget corresponding to global constraints
  */
case class ConstrainedMatchingBudget(budgetLocal: BSV[Double], budgetGlobal: BSV[Double])
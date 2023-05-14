package com.linkedin.dualip.data

/**
  * A MOO block of data. A vertical slice of design matrix, specifically the variables in the same simplex constraint sum x <= 1
  * We need to keep this data together because we need to do a simplex projection on it.
  *
  * Column (variable indices in "a" and "c" are relative, that is, variable is uniquely identified by
  * a combination of block id and internal id.
  *
  * internal representation is optimized for the operations that algorithm implements and data characteristics:
  * in particular, dense constraints matrix with few rows.
  *
  * @param id        - unique identifier of the block, i.e. impression id for some problems
  * @param a         - a dense constraints matrix a(row)(column)
  * @param c         - a dense objective function vector
  * @param problemId - unique identifier for distinguishing a specific LP problem
  */
case class MooData(id: Long, a: Array[Array[Double]], c: Array[Double], problemId: Long)

/**
  * A constraint block of data. A data point of constraint vector.
  *
  * @param row       - a specific row number of the constraint vector
  * @param value     - the corresponding constraint value for the row
  * @param problemId - unique identifier for distinguishing a specific LP problem
  */
case class ConstraintBlock(row: Int, value: Double, problemId: Long)

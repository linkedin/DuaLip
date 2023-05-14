package com.linkedin.dualip.data

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.projection.Projection

/**
  * case class for the dual variables corresponding to local and global constraints
  *
  * @param lambdaLocal  : dual variables corresponding to local constraints
  * @param lambdaGlobal : dual variables corresponding to global constraints
  */
case class ConstrainedMatchingDuals(lambdaLocal: Array[Double], lambdaGlobal: Array[Double])

/**
  * case class for the dual variables corresponding to local and global constraints
  *
  * @param lambdaLocal  : dual variables corresponding to local constraints in BSV format
  * @param lambdaGlobal : dual variables corresponding to global constraints in BSV format
  */
case class ConstrainedMatchingDualsBSV(lambdaLocal: BSV[Double], lambdaGlobal: BSV[Double])

/**
  * case class for constrained matching solver parameters
  *
  * @param constrainedMatchingDataPath : Path of A matrix, G matrix and c vector combined in a special data block
  * @param localBudgetPath             : Path of the budgets corresponding to local constraints
  * @param globalBudgetPath            : Path of the budgets corresponding to global constraints
  * @param format                      : The format of input data, e.g. avro or orc
  * @param numOfPartitions             : number of partitions for sp
  * @param enableHighDimOptimization   : enables high-dimensional optimization
  * @param numLambdaPartitions         : number of partitions for the duals
  */
case class ConstrainedMatchingSolverParams(
  constrainedMatchingDataPath: String,
  localBudgetPath: String,
  globalBudgetPath: String,
  format: DataFormat,
  numOfPartitions: Int,
  enableHighDimOptimization: Boolean = false,
  numLambdaPartitions: Option[Int] = None)


/**
  * case class for the A-G-c data block corresponding to the constrained matching problem
  *
  * @param id       : row-id
  * @param data     : c-A-G data corresponding to a given row-id
  * @param metadata : projection metadata
  *
  *                 The data field has the following structure.
  *
  *                 {
  *                 "name": "data",
  *                 "type": [
  *                 {
  *                 "type": "array",
  *                 "items": [
  *                 {
  *                 "type": "record",
  *                 "name": "data",
  *                 "fields": [
  *                 {
  *                 "name": "colId",
  *                 "type": "int"
  *                 },
  *                 {
  *                 "name": "c",
  *                 "type": "double"
  *                 },
  *                 {
  *                 "name": "A",
  *                 "type": "double"
  *                 },
  *                 {
  *                 "name": "G",
  *                 "type": "array",
  *                 "items": [
  *                 {
  *                 "fields": [
  *                 {
  *                 "name": "row-Id of G",
  *                 "type": "int"
  *                 },
  *                 {
  *                 "name": "value from G matrix",
  *                 "type": "double"
  *                 }
  *                 ]
  *                 }
  *                 ]
  *                 }
  *                 ]
  *                 }
  *                 ]
  *                 }
  *                 ]
  *                 }
  */
case class ConstrainedMatchingData(id: String, data: Seq[(Int, Double, Double, Seq[(Int, Double)])],
  metadata: Projection#Metadata = null)

object ConstrainedMatchingData {
  val optionalFields: Seq[String] = Seq("metadata")
}

/**
  * case class for the budgets
  *
  * @param budgetLocal  : budget corresponding to local constraints
  * @param budgetGlobal : budget corresponding to global constraints
  */
case class ConstrainedMatchingBudget(budgetLocal: BSV[Double], budgetGlobal: BSV[Double])
package com.linkedin.dualip.objective

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.data.MultiSlateMatchingData
import com.linkedin.dualip.util.DataFormat.DataFormat
import com.linkedin.dualip.util.VectorOperations.toBSV
import com.linkedin.dualip.util.{IOUtility, InputPaths}
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

import scala.util.Try

trait MultiSlateMatchingObjectiveLoader extends DualPrimalObjectiveLoader {

  /**
    * checks for consistency of the budget vectors
    *
    * @param budget : array of tuples of the form (Int, Double)
    */
  def checkBudget(budget: Array[(Int, Double)]): Unit = {
    println("checking the consistency of the budget data ..")
    val entityIds = budget.toMap.keySet
    budget.indices.foreach { i: Int =>
      require(entityIds.contains(i), f"$i index does not have a specified constraint")
    }
  }

  /**
    * loads budget vectors
    *
    * @param budgetPath : Path of the budgets
    * @param format     : The format of input data, e.g. avro or orc
    * @param spark      : spark session
    * @return
    */
  def loadBudgetData(budgetPath: String, format: DataFormat)
    (implicit spark: SparkSession): BSV[Double] = {

    println("loading budget data ..")
    val budget = IOUtility.readDataFrame(budgetPath, format)
      .map { case Row(_c0: Number, _c1: Number) => (_c0.intValue(), _c1.doubleValue()) }
      .collect
    checkBudget(budget)
    toBSV(budget, budget.length)
  }

  /**
    * loads A matrix and c vector combined in the multislateMatchingDataBlock
    *
    * @param multiSlateMatchingDataPath : Path of A matrix and c vector combined in a customized data block
    * @param format                     : The format of input data, e.g. avro or orc
    * @param spark                      : spark session
    * @return
    */
  def loadMultiSlateMatchingData(multiSlateMatchingDataPath: String, format: DataFormat)
    (implicit spark: SparkSession): Dataset[MultiSlateMatchingData] = {

    println("loading data for MultiSlate Matching solver ..")
    var multiSlateMatchingDataBlocks = IOUtility.readDataFrame(multiSlateMatchingDataPath, format)
      .toDF("id", "data", "metadata")

    MultiSlateMatchingData.optionalFields.foreach {
      field =>
        if (Try(multiSlateMatchingDataBlocks(field)).isFailure) {
          multiSlateMatchingDataBlocks = multiSlateMatchingDataBlocks.withColumn(field, lit(null))
        }
    }

    multiSlateMatchingDataBlocks
      .as[MultiSlateMatchingData]
      .repartition(spark.sparkContext.defaultParallelism)
      .persist(StorageLevel.MEMORY_ONLY)
  }

  /**
    * loads data for MultiSlate matching problem
    *
    * @param inputPaths : input paths for the ACblock and budget
    * @param spark      : spark session
    * @return
    */
  def loadData(inputPaths: InputPaths)
    (implicit spark: SparkSession): (Dataset[MultiSlateMatchingData], BSV[Double]) = {
    println("invoking loadData from Multislate matching solver ..")
    (loadMultiSlateMatchingData(inputPaths.ACblocksPath, inputPaths.format), loadBudgetData(inputPaths.vectorBPath,
      inputPaths.format))
  }
}

package com.linkedin.spark.common.lib

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

/**
  * Utilities of Spark local unit test
  */
object TestUtils {

  /**
    * Create the local SparkSession used for general-purpose spark unit test
    * end-to-end read/write solution unit test.
    *
    * @param appName: name of the local spark app
    * @param numThreads: parallelism of the local spark app
    * @param sparkConf: provide user specific Spark conf object rather than using default one. The appName and default
    *                   config in sparkConf will not be honored. User should set sparkConf and numThreads explicitly.
    */
  def createSparkSession(appName: String = "localtest", numThreads: Int = 4,
    sparkConf: SparkConf = new SparkConf()): SparkSession = {

    /*
     * Below configs are mimicking the default settings in our Spark cluster so user does not need to set them
     * in their prod jobs.
     * Expression Encoder config is to enable scala case class to understand avro java type as its field
     */
    val conf = sparkConf
    conf.set("spark.isolated.classloader", "true")
    conf.set("spark.isolated.classes", "org.apache.hadoop.hive.ql.io.CombineHiveInputFormat$CombineHiveInputSplit")

    val sessionBuilder = SparkSession.builder().appName(appName).master(s"local[$numThreads]").config(conf)

    val spark = sessionBuilder.getOrCreate()

    // Disable _SUCCESS file generation when writing data to local disk
    spark.sparkContext.hadoopConfiguration.set(
      "mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")

    spark
  }
}
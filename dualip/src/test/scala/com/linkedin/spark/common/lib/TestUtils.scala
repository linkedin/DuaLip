/*
 * BSD 2-CLAUSE LICENSE
 *
 * Copyright 2021 LinkedIn Corporation
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
 
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
    * @param sparkConf: provide user specific Spark conf object rather than using default one. The appName and master
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
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.isolated.classloader", "true")
    conf.set("spark.isolated.classes", "org.apache.hadoop.hive.ql.io.CombineHiveInputFormat$CombineHiveInputSplit")
    conf.set("spark.expressionencoder.org.apache.avro.specific.SpecificRecord",
      "com.databricks.spark.avro.AvroEncoder$")

    val sessionBuilder = SparkSession.builder().appName(appName).master(s"local[${numThreads}]").config(conf)

    val spark = sessionBuilder.getOrCreate()

    // Disable _SUCCESS file generation when writing data to local disk
    spark.sparkContext.hadoopConfiguration.set(
      "mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")

    spark
  }
}
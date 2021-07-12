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
 
package com.linkedin.dualip.util

import com.linkedin.dualip.util.DataFormat.{AVRO, DataFormat, JSON, ORC, CSV}
import java.io.{BufferedWriter, OutputStreamWriter, PrintWriter}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import scala.collection.mutable

/**
  * Utils to read input matrix and convert to BlockMatrix/BlockVector
  */
object IOUtility {

  /**
    * Read dataframe by Spark, currently support AVRO, ORC, JSON or CSV formats.
    *
    * @param inputPath   Path for the input files
    * @param inputFormat Input format, e.g. AVRO, ORC, JSON or CSV
    * @param spark Spark session
    * @return the read dataframe
    */
  def readDataFrame(
    inputPath: String,
    inputFormat: DataFormat
  )(implicit spark: SparkSession): DataFrame = {
    inputFormat match {
      case AVRO => spark.read.format(AVRO.toString).load(inputPath)
      case ORC => spark.read.format(ORC.toString).load(inputPath)
      case JSON => spark.read.format(JSON.toString).load(inputPath)
      case CSV => spark.read.format(CSV.toString).options(Map("inferSchema"->"true", "header"-> "true")).load(inputPath)
      case _ =>
        throw new IllegalArgumentException(s"Unknown format $inputFormat, " +
          s"use avro, orc, json or csv only")
    }
  }

  /**
   * Save dataframe to HDFS
   *
   * @param dataFrame The output dataframe
   * @param outputPath The output path
   * @param outputFormat The saved file format, either AVRO, ORC or JSON
   * @param numPartitions Optional number of partitions of output dataframe,
   *                      If not provided, partitioning is based on the upstream steps
   */
  def saveDataFrame(
    dataFrame: DataFrame,
    outputPath: String,
    outputFormat: DataFormat = AVRO,
    numPartitions: Option[Int] = None
  ): Unit = {
    val dataFrameWriter = { numPartitions match {
      case Some(partitions) => dataFrame.repartition(partitions)
      case _ => dataFrame
    }}
      .write
      .mode(SaveMode.Overwrite)

    outputFormat match {
      case AVRO => dataFrameWriter.format(AVRO.toString).save(outputPath)
      case ORC => dataFrameWriter.format(ORC.toString).save(outputPath)
      case JSON => dataFrameWriter.format(JSON.toString).save(outputPath)
      case _ =>
        throw new IllegalArgumentException(s"Unknown format $outputFormat, " +
          s"use avro, orc or json only")
    }
  }

  /**
    * Save log string to a text file
    *
    * @param log The log in string format
    * @param logPath The output path
    */
  def saveLog(log: String, logPath: String): Unit = {
    val fs = FileSystem.get(new Configuration())
    val outputStream = fs.create(new Path(logPath))
    val writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(outputStream)))
    try {
      writer.write(log)
    } finally {
      writer.close()
    }
  }

  /**
    * A generic timer function to time the block of code
    * @param block the function or block of code that needs to be timed
    * @param log an overall log buffer to write the time
    * @tparam R return type of the block of code being timed
    * @return
    */
  def time[R](block: => R, log: mutable.Map[String, String]): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    val time = (t1 - t0).toDouble / 1000000000
    log += ("time(sec)" -> f"$time%.3f")
    result
  }

  def iterationLog(log: mutable.Map[String, String]): String = {
    val state = log.map(item => f"${item._1}: ${item._2}").mkString("\t") + "\n"
    print(state)
    state
  }

  /**
    * Print the argument list
    * @param args argument list passed as an array of string.
    * @return
    */
  def printCommandLineArgs(args: Array[String]): Unit = {
    var i = 0
    for (i <- 0 until args.length by 2) {
      println(f"${args(i)} ${args(i + 1)}")
    }
  }
}


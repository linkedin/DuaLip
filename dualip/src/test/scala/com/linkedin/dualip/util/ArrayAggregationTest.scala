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

import com.linkedin.spark.common.lib.TestUtils
import org.apache.spark.sql.{Dataset, SparkSession}
import org.testng.Assert
import org.testng.annotations.Test

import scala.util.Random

class ArrayAggregationTest {
  import ArrayAggregation._
  @Test
  def testAggregateArrays(): Unit = {
    val l = Array(0.1, 0.2, 0.0, 0.5)
    val r = Array(0.3, 0.0, 0.0, 3.3)
    val expectedResult = Array(0.4, 0.2, 0.0, 3.8)
    val result = aggregateArrays(l, r)
    Assert.assertEquals(result.toVector, expectedResult.toVector) // convert to scala vectors because java Array compares by reference
  }

  @Test(
    expectedExceptions = Array(classOf[java.lang.IllegalArgumentException]),
    expectedExceptionsMessageRegExp = ".*Cannot aggregate arrays of different sizes"
  )
  def testAggregateArraysWithException(): Unit = {
    val l = Array(0.1, 0.2, 0.0)
    val r = Array(0.3, 0.0, 0.0, 3.3)
    aggregateArrays(l, r)
  }

  @Test
  def testPartitionBounds(): Unit = {
    // even split into partitions
    Assert.assertEquals(partitionBounds(arrayLength = 8, numPartitions = 2, partition = 1), (4,8))
    // uneven split, larger partitions should be stacked first
    Assert.assertEquals(partitionBounds(arrayLength = 8, numPartitions = 3, partition = 0), (0,3))
    Assert.assertEquals(partitionBounds(arrayLength = 8, numPartitions = 3, partition = 1), (3,6))
    Assert.assertEquals(partitionBounds(arrayLength = 8, numPartitions = 3, partition = 2), (6,8))
  }

  @Test(
    expectedExceptions = Array(classOf[java.lang.IllegalArgumentException])
  )
  def testPartitionBoundsWithException(): Unit = {
    // requesting higher partition than total number of partitions
    Assert.assertEquals(partitionBounds(arrayLength = 3, numPartitions = 2, partition = 2), (4,8))
  }

  @Test
  def testPartitionArray(): Unit = {
    val input = (0 until 90).map(_.toDouble)
    // two variants of partitioning: evenly sized partititions and unevenly sized partitions
    for(partitions <- Seq(10, 17)){
      val out: Seq[(Int, Array[Double])] = partitionArray(input.toArray, partitions)
      // check that we can assemble to the original input
      val combinedOut: Seq[Double] = out.sortBy { case (partition, subarray) => partition }
        .map { case (partition, subarray) => subarray }
        .fold(Array[Double]())(_ ++ _).toSeq
      Assert.assertEquals(combinedOut, input)
    }
  }

  /**
    * Profiling section, test is disabled. Generates large number of random arrays and aggregates
    * them using naive spark approach and ArrayAggregation functions. Both methods are wrapped
    * into a spark job to ensure fair overheads.
    */
  //@Test
  def profileArraysAggregation(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    import spark.implicits._
    val n = 1000  // number of arrays
    val d = 10000 // array dimensionality
    spark.sparkContext.setLogLevel("warn")

    val data: Seq[Array[Double]] = (0 until n).map { _ =>
      (0 until d).toArray.map(_ => Random.nextDouble())
    }
    val dataFlattened = data.flatMap(arr => arr.zipWithIndex.map { case (a,b) => (b,a)})
    val datasetFlattened: Dataset[(Int, Double)] = spark.createDataset(dataFlattened)
    val datasetOfArrays: Dataset[Array[Double]] = spark.createDataset(data)

    println("Generated random data")
    val t0 = System.currentTimeMillis()
    datasetOfArrays.reduce(aggregateArrays(_, _))
    val t1 = System.currentTimeMillis()
    println(s"ArraysAggregation elapsed time: ${(t1 - t0)/1000.0} seconds")

    datasetFlattened.rdd.reduceByKey(_ + _).collect()
    val t2 = System.currentTimeMillis()
    println(s"Naive spark elapsed time: ${(t2 - t1)/1000.0} seconds")
  }
}

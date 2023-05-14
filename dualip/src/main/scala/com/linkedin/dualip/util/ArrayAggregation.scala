package com.linkedin.dualip.util

import java.util

/**
  * Util methods for fast aggregation of very large arrays. Intended for very high dimensional lambdas.
  * For example, 1M-dimensional. Each executor will likely produce dense vector with almost all dimensions.
  * Standard approach where we convert gradients into (key, value) pairs and use spark aggregation is slow.
  * This object provides methods to do array summation, as well as array partitioning to aggregate arrays
  * in a distributed way (sending all arrays to the driver would kill driver I/O).
  *
  * Profiling on a single core (see ArrayAggregationTest) has the following results for
  * aggregating 1000 arrays of 10,000 dimension:
  * baseline:      56-58 sec
  * this library:  1.2 sec (most of it is spark overhead, if the method is executed directly it only takes 0.03 sec).
  *
  * There is also an I/O win, arrays of doubles are stored with minimal overheads.
  */
object ArrayAggregation {
  /**
    * Aggregate two arrays of the same size by allocating new array and then looping over dimensions filling
    * the sums. This is the fastest method that I found so far. Other alternatives that I also tried and
    * that are much slower are:
    *   - summation of Breeze Sparse/Dense vectors
    *   - summation of maps (flattened representation)
    *   - summation using scala collections
    *
    * Key differentiators:
    *   - use of Array[Double] which is a wrapper of java double[]. It has special implementation for primitive types
    *     that native scala collections do not support
    *   - looping using while loop, scala functional transformations (zip, map etc) are much slower
    *
    * @param l
    * @param r
    * @return
    */
  def aggregateArrays(l: Array[Double], r: Array[Double]): Array[Double] = {
    require(l.length == r.length, "Cannot aggregate arrays of different sizes")
    val result = new Array[Double](l.size)
    var i = 0
    while (i < l.size) {
      result(i) = l(i) + r(i)
      i += 1
    }
    result
  }

  /**
    * Method to find [start, end) positions in the array of a given partition.
    * Array is partitioned into roughly identical partitions, the length of a partition may differ by one.
    *
    * @param arrayLength
    * @param numPartitions
    * @param partition
    * @return
    */
  def partitionBounds(arrayLength: Int, numPartitions: Int, partition: Int): (Int, Int) = {
    require(partition < numPartitions, s"partition number cannot be larger than total number of partitions")
    require(numPartitions <= arrayLength)
    // some partitions will have size basePartitionSize and some basePartitionSize+1
    // we will stack larger partitions in the beginning of the array
    val basePartitionSize = arrayLength / numPartitions
    val numLargerPartitions = arrayLength - basePartitionSize * numPartitions
    // second term accounts for larger partitions stacked prior to partition in question
    val startIndex = partition * basePartitionSize + math.min(partition, numLargerPartitions)
    val partitionSize = if (partition < numLargerPartitions) {
      basePartitionSize + 1
    } else {
      basePartitionSize
    }
    val endIndex = startIndex + partitionSize
    (startIndex, endIndex)
  }

  /**
    * Partitions array into numPartitions of almost equal size. Sizes differ at most by 1 because
    * array size may not be divisible by numPartitions. All larger partitions go first.
    * E.g. array of size 8 partitioned into 3will have subarrays of sizes 3, 2, 2
    *
    * @param data          - input array
    * @param numPartitions - num partitions
    * @return - tuples of (partitionNumber, sub-array)
    */
  def partitionArray(data: Array[Double], numPartitions: Int): Seq[(Int, Array[Double])] = {
    require(numPartitions <= data.length)
    (0 until numPartitions).map { partition =>
      val (start, end) = partitionBounds(data.length, numPartitions, partition)
      (partition, util.Arrays.copyOfRange(data, start, end))
    }
  }
}

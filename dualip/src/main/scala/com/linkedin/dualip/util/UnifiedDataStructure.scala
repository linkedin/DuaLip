package com.linkedin.dualip.util

import org.apache.spark.sql.{Dataset, Encoder}

import scala.reflect.ClassTag

/**
  * New data types to support objects being passed and computed as either Array or Dataset
  */
trait MapReduceCollectionWrapper[A] {
  val value: Object

  /*
   * todo: encoder is only needed for MapReduceDataset, but not for MapReduceArray
   *       and this API requirement is excessive. Figure out if we can refactor
   *       the code to only initialize/pass the encoder when it is needed
   */
  def map[B: ClassTag](op: A => B, encoder: Encoder[B]): MapReduceCollectionWrapper[B]

  def reduce(op: (A, A) => A): A

  def isEmpty: Boolean
}

case class MapReduceArray[A](value: Array[A]) extends MapReduceCollectionWrapper[A] {
  override def map[B: ClassTag](op: A => B, encoder: Encoder[B]): MapReduceArray[B] = {
    MapReduceArray[B](value.map(op))
  }

  override def reduce(op: (A, A) => A): A = {
    value.reduce(op)
  }

  def isEmpty: Boolean = {
    value.isEmpty
  }
}

case class MapReduceDataset[A](value: Dataset[A]) extends MapReduceCollectionWrapper[A] {
  override def map[B: ClassTag](op: A => B, encoder: Encoder[B]): MapReduceDataset[B] = {
    MapReduceDataset[B](value.map(op)(encoder))
  }

  override def reduce(op: (A, A) => A): A = {
    value.reduce(op)
  }

  def isEmpty: Boolean = {
    value.isEmpty
  }
}

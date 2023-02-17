package com.linkedin.dualip.util

import org.apache.spark.sql.{Dataset, Encoder}
import scala.reflect.ClassTag

/**
 * New data types to support objects being passed and computed as either Array or Dataset
 */
trait MapReduceCollectionWrapper[A] {
  val value: Object
  def map[B: ClassTag](op: A => B, encoder: Encoder[B]): MapReduceCollectionWrapper[B]
  def reduce(op: (A, A) => A): A
}

case class MapReduceArray[A](value: Array[A]) extends MapReduceCollectionWrapper[A] {
  override def map[B: ClassTag](op: A => B, encoder: Encoder[B]): MapReduceArray[B] = {
    MapReduceArray[B](value.map(op))
  }
  override def reduce(op: (A, A) => A ): A = {
    value.reduce(op)
  }
}

case class MapReduceDataset[A](value: Dataset[A]) extends MapReduceCollectionWrapper[A] {
  override def map[B: ClassTag](op: A => B, encoder: Encoder[B]): MapReduceDataset[B] = {
    MapReduceDataset[B](value.map(op)(encoder))
  }
  override def reduce(op: (A, A) => A ): A = {
    value.reduce(op)
  }
}

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

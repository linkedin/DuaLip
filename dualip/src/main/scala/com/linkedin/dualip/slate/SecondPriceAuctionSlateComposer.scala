package com.linkedin.dualip.slate

import com.linkedin.dualip.data.MatchingData

import scala.collection.mutable.ArrayBuffer

/**
  * Second price slate optimizer component
  *
  * @param slateSize
  */
class SecondPriceAuctionSlateComposer(slateSize: Int) extends SlateComposer with Serializable {
  /**
    * Slate optimizer API implementation, simply bridges slate API with DP solver
    * currently only returns one slate
    *
    * @param block  - data block, usually all parameters related to individual impression
    * @param lambda - vector of dual variables (note the use of java array for performance
    * @return generated slate
    */
  override def getSlate(block: MatchingData, lambda: Array[Double]): Seq[Slate] = {
    val origIds = block.data.map(_._1).toArray
    val subC = block.data.map(_._2).toArray
    val subLambda = origIds.map(id => lambda(id))
    val (ids, costs, objective) = SecondPriceAuctionSlateComposer.secondPriceSolver(subC, subLambda, slateSize)
    //remap ids to the original ones
    Seq(Slate(1.0, objective, ids.map(id => origIds(id)).zip(costs)))
  }
}

/**
  * Second price Dynamic programming slate optimizer
  */
object SecondPriceAuctionSlateComposer {

  /**
    * Second price auction slate optimization:
    * find the sequence (ordered) of elements that form a slate i1, i2, ... ik
    * that maximizes the objective:
    * c[i2] * (1 - lambda[i1]) + ... + c[im] * (1 - lambda[i(m-1)] + ...
    *
    * We use backwards induction algorithm to solve this Dynamic Programming problem.
    *
    * Optimization.
    * The algorithm spends most of this time in this method, so it is important to optimize it.
    * While we cannot improve complexity O(c.length * c.length * k), optimizing the constant factor can save a lot,
    * as we solve a huge number of relatively small problems.
    * c.length ~ hundreds (number of candidates)
    * k = 20 for ads use case - this is the size of the slate that we need to fill
    * Some of critical optimizations that help:
    *   - use of arrays instead of other data structures (e.g. immutable vectors)
    *   - while loop instead of for loop (which is slow in scala) - especially for internal loops
    *
    * @todo Figure out payment rule for the last element in the slate, it might depend
    *       on the business logic and hence affect the algorithm.
    *       Currenly the last element has zero value.
    * @param c         - vector of expected utilities of eligible items, we expect it to be sorted in decreasing order
    * @param lambda    - dual variables that correspond to the eligible items
    * @param slateSize - corresponds to "k" above, amount of positions to fill.
    * @return (the vector of selected ids, reduced cost of the slate)
    */
  def secondPriceSolver(c: Array[Double], lambda: Array[Double], slateSize: Int): (Array[Int], Array[Double], Double) = {
    // number of non trivial states of DP, the state at step k is defined by the ad at step k
    // there is an extra "no-ad" state that we don't keep explicitly
    val n = c.length
    val maxSlateSize = Math.min(n, slateSize) // cannot put more items than candidates

    // check input
    for (i <- 0 until n - 1) {
      if (c(i) < c(i + 1)) throw new Exception("secondPriceSolver: input data is not sorted")
    }

    // represents residual value at state i, we initialize with 0, meaning the last ad pays zero (can be some constant, does not impact the optimization).
    var nextStateResiduals: Array[Double] = Array.fill[Double](n)(0)
    // similarly, represents the current state
    var currentStateResiduals: Array[Double] = Array.fill[Double](n)(0)
    // forward links to reconstruct the back propagation solution
    // represents the best move for each state at each step, last step terminates the sequence
    val bestMoves = Array.ofDim[Int](maxSlateSize - 1, n)

    //backwards induction algorithm

    for (pos <- maxSlateSize - 2 to 0 by -1) { // iterate over positions of the slate, we start from second to last
      var state = pos // minimum number of items is required to fill previous positions, so we do not start from 0.
      while (state < n) { // iterate over possible states in this step
        // find the best move, note that we cannot violate the order of ads, which restricts the number
        // of possible moves (ads). That is, after ad "i" we can only put ad "i+1" or higher rank.
        currentStateResiduals(state) = 0 // best value so far, 0.0 is always achievable by not displaying any ads
        bestMoves(pos)(state) = -1 // special sequence terminating index.

        var move = state + 1 // next items must have higher index than current item
        var bestValue = 0.0
        var residual = 0.0
        while (move < n) {
          residual = c(move) * (1.0 - lambda(state)) + nextStateResiduals(move)
          if (residual > bestValue) {
            currentStateResiduals(state) = residual
            bestValue = residual
            bestMoves(pos)(state) = move
          }
          move += 1
        }
        state += 1
      }
      // flip current and next state vectors
      val tmp = nextStateResiduals
      nextStateResiduals = currentStateResiduals
      currentStateResiduals = tmp
    }

    // now reconstruct the best path
    var move = -1
    var position = 0
    var bestValue = 0.0
    val result = new ArrayBuffer[Int]
    val costs = new ArrayBuffer[Double]
    var state = 0
    while (state < n) {
      if (nextStateResiduals(state) > bestValue) {
        move = state
        bestValue = nextStateResiduals(state)
      }
      state += 1
    }

    while (move != -1) {
      result += move
      move = if (position < maxSlateSize - 1) bestMoves(position)(move) else -1
      if (move != -1) {
        costs += c(move)
      } else {
        costs += 0
      }
      position += 1
    }
    (result.toArray, costs.toArray, bestValue)
  }
}
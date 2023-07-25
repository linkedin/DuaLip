package com.linkedin.dualip.data

/**
  * Representation of the budget vector/budget per item id for Matching Problem.
  *
  * @param itemId - id of the item (i.e. product id)
  * @param budget - the budget for itemId/the b(itemId) value in the budget (b) vector
  */
case class BudgetData(itemId: Int, budget: Double)

/**
  * Representation of the budget for a specific entity, given a constraint matrix.
  * This is used to represent the budget of an entity that may have several constraints, as in the
  * Multiple Matching Problem.
  *
  * @param entityIndex - index for the entity
  * @param constraintIndex - index of the constraint
  * @param budgetValue - budget for this entity-constraint combo, or b(entityIndex)(constraintIndex)
  */
case class MultipleMatchingBudgetData(entityIndex: Int, constraintIndex: Int, budgetValue: Double)

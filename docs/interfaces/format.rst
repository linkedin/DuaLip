Input Format
----------------

Users are expected to provide the A, b, and c of the LP formulation to the solver. The solver takes input in the following format:

For vector b, the input should contain the budget for each item. The representation needs to be dense: every item needs to have a budget.
For input in AVRO, the schema should be as follows:

.. code:: json

	{
	   "name" : "itemId",
	   "type" : "int"
	 }, {
	   "name" : "budget",
	   "type" : "double"
	 }

For matrix A and vector c, if it is a MOO problem, we take a dense representation.

* id is a unique identifier of the block.
* a contains the dense constraints matrix A(row)(column).
* c contains a dense objective function vector.

For input in AVRO, the schema should be as follows:

.. code:: json

	{
	    "name" : "id",
	    "type" : [ "long" ]
	  }, {
	    "name" : "a",
	    "type" : [ {
	      "type" : "array",
	      "items" : [ {
	        "type" : "array",
	        "items" : [ "double"]
	      }]
	    }]
	  }, {
	    "name" : "c",
	    "type" : [ {
	      "type" : "array",
	      "items" : "double"
	    }]
	  }

If it is a Matching problem, we take a sparse representation. We keep the items in A and c which belong to same block together, to facilitate projection.

id is a unique identifier of the block, i.e. impression id for some problems. Each id correspond to an array of tuples, which is in the format of
(rowId, c(rowId), a(rowId)).

.. code:: json

	{
	    "name" : "id",
	    "type" : [ "string" ]
	  }, {
	    "name" : "data",
	    "type" : [ {
	      "type" : "array",
	      "items" : [ {
	        "type" : "record",
	        "name" : "data",
	        "fields" : [ {
	          "name" : "rowId",
	          "type" : "int"
	        }, {
	          "name" : "c",
	          "type" : "double"
	        }, {
	          "name" : "a",
	          "type" : "double"
	        } ]
	      } ]
	    } ]
	  }
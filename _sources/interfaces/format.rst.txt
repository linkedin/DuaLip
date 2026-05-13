Input Format
----------------
Users are expected to provide the tensors :math:`A`, :math:`b`, and :math:`C` defining the LP formulation to the solver. In the matching problem implementation, both :math:`A` and :math:`C` are assumed to be stored as CSC sparse matrices for memory efficiency.
See `here <https://docs.pytorch.org/docs/stable/generated/torch.sparse_csc_tensor.html>`_ for details on the CSC format in PyTorch.
The vector :math:`b`, however, is represented as a dense vector.

The CSC sparse format exploits both structured (e.g., block-diagonal or triangular) and unstructured sparsity patterns in the :math:`A` and :math:`C` matrices, significantly reducing memory footprint and improving computational efficiency at scale.

In addition, users must specify the constraint sets :math:`\mathcal{C}_i` for each user in the form of **projection maps**. Projection maps are dictionaries that associate a unique constraint identifier (key) with a data structure describing the indices belonging to that constraint set, along with the parameters defining the constraint :math:`\mathcal{C}_i`.

For example, a projection map for a closed simplex constraint may be defined as follows:

.. code-block:: python

   {
       "simplex_with_radius_one": ProjectionEntry(
           indices=[0, 1, 2],
           proj_type="simplex",
           proj_params=1.0  # radius of the simplex
       )
   }

See :ref:`The DuaLip Solver <solver>` section for a complete list of supported constraint sets.

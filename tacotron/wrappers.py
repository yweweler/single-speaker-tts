import tensorflow as tf
# from tensorflow import contrib as tfc
import tensorflow.contrib as tfc


class PrenetWrapper(tfc.rnn.RNNCell):
    def __init__(self):
        super().__init__()

    @property
    def state_size(self):
        pass

    @property
    def output_size(self):
        pass

    def compute_output_shape(self, input_shape):
        pass


class InputProjectionWrapper(RNNCell):
  """Operator adding an input projection to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your inputs in time,
  do the projection on this batch-concatenated sequence, then split it.
  """

  def __init__(self,
               cell,
               num_proj,
               activation=None,
               input_size=None,
               reuse=None):
    """Create a cell with input projection.

    Args:
      cell: an RNNCell, a projection of inputs is added before it.
      num_proj: Python integer.  The dimension to project to.
      activation: (optional) an optional activation function.
      input_size: Deprecated and unused.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
    super(InputProjectionWrapper, self).__init__(_reuse=reuse)
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    if not _like_rnncell(cell):
      raise TypeError("The parameter cell is not RNNCell.")
    self._cell = cell
    self._num_proj = num_proj
    self._activation = activation
    self._linear = None

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def call(self, inputs, state):
    """Run the input projection and then the cell."""
    # Default scope: "InputProjectionWrapper"
    if self._linear is None:
      self._linear = _Linear(inputs, self._num_proj, True)
    projected = self._linear(inputs)
    if self._activation:
      projected = self._activation(projected)
    return self._cell(projected, state)

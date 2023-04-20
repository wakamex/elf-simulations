"""Define Python user-defined exceptions."""


class InvalidCheckpointTimeError(Exception):
    """If the checkpoint time isn't divisible by the checkpoint duration or is in the future, it's an
    invalid checkpoint and we should revert.
    """


class OutputLimitError(Exception):
    """If the output requirement is not met.  Often this is a minimum amount out as slippage protection."""


class UnsupportedOptionError(Exception):
    """If the output requirement is not met.  Often this is a minimum amount out as slippage protection."""


class DivisionByZeroError(Exception):
    """For FixedPoint type; thrown if trying to divide any FixedPoint number by zero."""

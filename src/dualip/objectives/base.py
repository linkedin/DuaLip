from abc import ABC, abstractmethod
from dataclasses import dataclass

from dualip.types import ObjectiveResult


@dataclass
class BaseInputArgs(ABC):
    """
    Abstract base class for input arguments to objective functions.
    Each objective type should extend this with its specific requirements.
    """

    def __post_init__(self):
        """Validate common fields after initialization."""
        pass


class BaseObjective(ABC):
    """
    Abstract base class for objective functions.
    """

    @abstractmethod
    def calculate(self) -> ObjectiveResult:
        pass

    def set_gamma(self, gamma: float) -> None:
        """Update the regularization parameter. Override in subclasses that use gamma."""
        pass

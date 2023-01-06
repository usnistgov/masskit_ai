from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module
import torch.nn.functional as functional

"""
base losses

"""
class BaseLoss(Module, ABC):
    """
    abstract base class for losses
    loss is implemented as a pytorch module
    """

    def __init__(self, config=None, set=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

    @abstractmethod
    def forward(self, output, batch, params=None) -> Tensor:
        """
        calculate the loss

        :param output: output dictionary from the model
        :param batch: batch data from the dataloader
        :param params: optional dictionary of parameters, such as epoch type
        :return: loss tensor
        """
        pass


class MSELoss(BaseLoss):
    """
    mean square error
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, output, batch, params=None) -> Tensor:
        return_val = functional.mse_loss(output.y_prime, batch.y)
        return return_val


class L1Loss(BaseLoss):

    """
    l1 loss
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, output, batch, params=None) -> Tensor:
        return_val = functional.l1_loss(output.y_prime, batch.y)
        return return_val


class MSEKLLoss(BaseLoss):
    """
    mean square error plus kl divergence
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, output, batch, params=None) -> Tensor:
        return_val = functional.mse_loss(output.y_prime, batch.y) + \
                     output.score / self.config.input[params['loop']].num
        return return_val

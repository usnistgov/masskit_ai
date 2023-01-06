from typing import Tuple
import torch
from torch import Tensor
import torch.nn.functional as functional
from abc import abstractmethod
from masskit_ai.base_losses import BaseLoss
from masskit_ai.spectrum.peptide.peptide_constants import EPSILON
"""
Various losses for predicting spectra

- the "output" from the model is a dictionary
  - output.y_prime contains a batch of predicted spectra
- "batch" is the input to the model
  - batch.y is a batch of experimental spectra corresponding to the predicted spectra
- each batch of spectra is a float 32 tensor of shape (batch, channel, mz_bins)
  - by convention, channel 0 are intensities, which are not necessarily scaled
  - channel 1 are standard deviations of the corresponding intensities
"""

class BaseSpectrumLoss(BaseLoss):
    """
    abstract base class for spectrum losses
    assumes spectra have dimensions (batch, channel, mz_array)
    """
    def __init__(self, intensity_channel=0, variance_channel=1, epsilon=EPSILON,
                 *args, **kwargs) -> None:
        """
        init BaseSpectrumLoss

        :param intensity_channel: which channel to operate on
        :param variance_channel: the channel in the predicted spectra that has the predicted variance per peak
        :param epsilon: small value, used in division, etc.
        """
        super(BaseSpectrumLoss, self).__init__(*args, **kwargs)
        self.intensity_channel = intensity_channel
        self.variance_channel = variance_channel
        self.epsilon = epsilon

    def extract_spectra(self, output, batch) -> Tuple[Tensor, Tensor]:
        predicted_spectrum = output.y_prime[:, self.intensity_channel:self.intensity_channel + 1, :]
        true_spectrum = batch.y
        if self.config.ml.loss.sqrt_intensity:
            predicted_spectrum[predicted_spectrum < 0.0] = 0.0
            predicted_spectrum = predicted_spectrum.sqrt()
            true_spectrum[true_spectrum < 0.0] = 0.0
            true_spectrum = true_spectrum.sqrt()
        return predicted_spectrum, true_spectrum

    def extract_variance(self, input_tensor: Tensor) -> Tensor:
        return input_tensor[:, self.variance_channel:self.variance_channel + 1, :]

    @abstractmethod
    def forward(self, output, batch, params=None) -> Tensor:
        """
        calculate the loss

        :param output: output dictionary from the model, type ModelOutput
        :param batch: batch data from the dataloader, type ModelInput
        :param params: optional dictionary of parameters, such as epoch type
        :return: loss tensor
        """
        pass


class SpectrumMSELoss(BaseSpectrumLoss):
    """
    mean square error of intensity channel
    """

    def __init__(self, *args, **kwargs) -> None:
        super(SpectrumMSELoss, self).__init__(*args, **kwargs)

    def forward(self, output, batch, params=None) -> Tensor:
        predicted_spectrum, true_spectrum = self.extract_spectra(output, batch)
        return functional.mse_loss(predicted_spectrum, true_spectrum)


class SpectrumMSEKLLoss(BaseSpectrumLoss):
    """
    mean square error of intensity channel plus KL divergence
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, output, batch, params=None) -> Tensor:
        predicted_spectrum, true_spectrum = self.extract_spectra(output, batch)
        return functional.mse_loss(predicted_spectrum, true_spectrum) + \
            output.score / self.config.input[params['loop']].num


class SpectrumCosineLoss(BaseSpectrumLoss):
    """
    cosine similarity of intensity channel
    """

    def __init__(self, *args, **kwargs) -> None:
        super(SpectrumCosineLoss, self).__init__(*args, **kwargs)

    def forward(self, output, batch, params=None) -> Tensor:
        predicted_spectrum, true_spectrum = self.extract_spectra(output, batch)
        return -functional.cosine_similarity(predicted_spectrum, true_spectrum, dim=-1).mean()


class SpectrumCosineKLLoss(BaseSpectrumLoss):
    """
    cosine similarity of intensity channel and KL divergence
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, output, batch, params=None) -> Tensor:
        predicted_spectrum, true_spectrum = self.extract_spectra(output, batch)
        return -functional.cosine_similarity(predicted_spectrum, true_spectrum, dim=-1).mean() + \
            output.score / self.config.ml.batch_size #self.config.input[params['loop']].num


class SpectrumLogCosineKLLoss(BaseSpectrumLoss):
    """
    log of cosine similarity of intensity channel and KL divergence
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, output, batch, params=None) -> Tensor:
        predicted_spectrum, true_spectrum = self.extract_spectra(output, batch)
        return -(functional.cosine_similarity(predicted_spectrum, true_spectrum, dim=-1).mean()).log() + \
                output.score * self.config.ml.model.FlipyFlopy.klw # bayesian-torch: batch_size, Tensorflow: self.config.input[params['loop']].num


class SpectrumNormalNLL(BaseSpectrumLoss):
    """
    negative log likelihood loss for a normal distribution for a spectral model that emits predictions
    and variance of that prediction
    omits constants in log likelihood
    """

    def __init__(self, *args, **kwargs) -> None:
        super(SpectrumNormalNLL, self).__init__(*args, **kwargs)

    def forward(self, output, batch, params=None) -> Tensor:
        predicted_spectrum, true_spectrum = self.extract_spectra(output, batch)
        diff = batch.y - output.y_prime
        variance = self.extract_variance(output.y_prime)
        if torch.isnan(variance).any():
            raise FloatingPointError("variance contains a NaN")
        return torch.mean(variance.log()) + torch.mean(diff.square() / (variance + self.epsilon))

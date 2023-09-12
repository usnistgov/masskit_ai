from abc import ABC, abstractmethod
import importlib
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from masskit.utils.general import class_for_name
from masskit_ai.lightning import seed_worker


class BaseSpectrumLightningModule(pl.LightningModule, ABC):
    """
    base class for pytorch lightning module used to run the training
    """

    def __init__(
            self, config=None, *args, **kwargs
    ):
        """
        :param config: configuration dictionary
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(config)
        self.config = config

        model_class = class_for_name(self.config.paths.modules.models, list(self.config.ml.model.keys())[0])
        self.model = model_class(config)

        self.loss_function = class_for_name(self.config.paths.modules.losses,
                                            self.config.ml.loss.loss_function
                                            )(config=self.config)

        self.validation_step_outputs = []
        self.test_step_outputs = []
        # create a list of metrics
        # for training
        self.train_metrics = {}
        # for validation and test
        self.valid_metrics = {}
        if "metrics" in self.config.ml and self.config.ml.metrics is not None:
            for metric in self.config.ml.metrics:
                self.train_metrics[metric] = class_for_name(self.config.paths.modules.metrics,
                                                            metric)(config=self.config)
                self.valid_metrics[metric] = class_for_name(self.config.paths.modules.metrics,
                                                            metric)(config=self.config)
        if "valid_metrics" in self.config.ml and self.config.ml.valid_metrics is not None:
            for metric in self.config.ml.valid_metrics:
                self.valid_metrics[metric] = class_for_name(self.config.paths.modules.metrics,
                                                            metric)(config=self.config)
        if "train_metrics" in self.config.ml and self.config.ml.train_metrics is not None:
            for metric in self.config.ml.train_metrics:
                self.train_metrics[metric] = class_for_name(self.config.paths.modules.metrics,
                                                            metric)(config=self.config)

    def forward(self, x):
        return self.model(x)

    def calc_loss(self, output, batch, params=None):
        """
        overrideable loss function

        :param output: output from the model
        :param batch: batch data, including input and true spectra
        :param params: optional dictionary of parameters, such as epoch type
        :return: loss
        """
        return self.loss_function(output, batch, params=params)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        """
        validation step

        :param batch: batch data tensor
        :param batch_idx: the index of the batch
        :param dataloader_idx: which dataloader is being used (None if just one)
        :return: loss
        """
        return self.validation_test_step(batch, batch_idx, 'valid', self.validation_step_outputs)

    def test_step(self, batch, batch_idx):
        return self.validation_test_step(batch, batch_idx, 'test', self.test_step_outputs)

    def configure_optimizers(self):
        optimizer = getattr(
            importlib.import_module('torch.optim'),
            self.config.ml.optimizer.optimizer_function
            )(self.model.parameters(), lr=self.config.ml.optimizer.lr)
        # scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=8)
        # scheduler =  MultiStepLR(optimizer, milestones=[25,50,75,100,125,150,175,200], gamma=0.33)

        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "monitor": "val_loss",
            #     },
            }

    # lightning appears to be in the process of redoing logging, and this function will
    # throw an exceptions with some versions of lightning
    # def custom_histogram_adder(self):
    #     # iterating through all parameters
    #     # lightning sometimes add the tensorboard logger itself, so handle this
    #     try:
    #         loggers = iter(self.logger)
    #     except TypeError:
    #         loggers = iter([self.logger])

    #     for logger in loggers:
    #         if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
    #             for name, params in self.model.named_parameters():
    #                 logger.experiment.add_histogram(name, params, self.current_epoch)

    def training_step_end(self, outputs):
        if isinstance(outputs, torch.Tensor):
            loss = outputs
        else:
            loss = torch.stack(outputs).mean()
        if self.global_step % 100 == 0:
            self.log("loss", loss, on_step=True)
        # self.logger.log_metrics({"loss": loss.item()})
        return loss

    def on_train_epoch_end(self):
        # progress_bar_dict was removed from lightning
        # output = f"Epoch {self.trainer.current_epoch}: train loss={float(self.trainer.progress_bar_dict['loss']):g}"
        # self.trainer.progress_bar_callback.main_progress_bar.write(output)
        # logging histograms, broken for some reason in more recent versions of lightning
        # self.custom_histogram_adder()
        for metric, metric_function in self.train_metrics.items():
            metric_function.reset()

    def validation_test_epoch_end(self, outputs, loop):
        """
        shared code between validation and test epoch ends.  logs and prints losses, resets metrics

        :param outputs: output list from model
        :param loop: 'val' or 'test'
        """
        # single validation set returns a list of tensors, multi validation returns a list of list of tensors
        # if single validation, insert it into a list
        if not isinstance(outputs[0], list):
            outputs = [outputs]
        losses = []
        for i, output in enumerate(outputs):
            losses.append(torch.tensor(output).mean().item())
        self.log(f'{loop}_loss', losses[0], prog_bar=True)
        for i in range(1, len(losses)):
            self.log(f'{loop}_loss_{i}', losses[i], prog_bar=True)
        output = f"Epoch {self.trainer.current_epoch}: {loop} loss={','.join([f'{x:g}' for x in losses])}"
        # self.trainer.progress_bar_callback.main_progress_bar.write(output)
        for metric, metric_function in self.valid_metrics.items():
            metric_function.reset()

    def on_validation_epoch_end(self):
        self.validation_test_epoch_end(self.validation_step_outputs, 'val')
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        self.validation_test_epoch_end(self.test_step_outputs, 'test')
        self.test_step_outputs.clear()

    @abstractmethod
    def validation_test_step(self, batch, batch_idx, loop, outputs):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass


class SpectrumLightningModule(BaseSpectrumLightningModule):
    """
    pytorch lightning module used to run the training
    """

    def __init__( self, config=None, *args, **kwargs):
        """
        :param config: configuration dictionary
        It's important that the config parameter is explictly defined so lightning serialization works
        """
        super().__init__(config=config, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        if self.config.ml.bayesian_network.bayes:
            loss = 0.0
            for step in range(self.config.ml.bayesian_network.sample_nbr):
                output = self.model(batch)
                loss += self.calc_loss(output, batch, params={'loop': 'train'})
            loss /= self.config.ml.bayesian_network.sample_nbr
        else:
            output = self.model(batch)
            loss = self.calc_loss(output, batch, params={'loop': 'train'})

        for metric, metric_function in self.train_metrics.items():
            # required to evaluate metric in each batch
            metric_value = metric_function(output, batch)
            self.log(f'training_' + metric, metric_value, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_test_step(self, batch, batch_idx, loop, outputs):
        """
        step shared with test and validation loops

        :param batch: batch
        :param batch_idx: index into data for batch
        :param loop: the name of the loop
        :param outputs: the list containing outputs
        :return: loss
        """
        if self.config.ml.bayesian_network.bayes:
            loss = 0.0
            for step in range(self.config.ml.bayesian_network.sample_nbr):
                output = self.model(batch)
                # outputs.append(output.y_prime)
                loss += self.calc_loss(output, batch, params={'loop': loop})
            loss /= self.config.ml.bayesian_network.sample_nbr
        else:
            output = self.model(batch)
            # outputs.append(output.y_prime)
            loss = self.calc_loss(output, batch, params={'loop': loop})
        outputs.append(loss)

        for metric, metric_function in self.valid_metrics.items():
            # required to evaluate metric in each batch
            metric_value = metric_function(output, batch)
            self.log(f'{loop}_' + metric, metric_value, prog_bar=True, on_step=False, on_epoch=True)
        return loss



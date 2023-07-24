#!/usr/bin/env python
import os
from datetime import date
import pytorch_lightning as pl
from masskit.utils.general import class_for_name
from masskit_ai.lightning import setup_datamodule
from masskit_ai.loggers import filter_pytorch_lightning_warnings, MSMLFlowLogger
from masskit_ai.spectrum.spectrum_lightning import SpectrumLightningModule
from omegaconf import DictConfig, open_dict
import hydra
from masskit_ai.callbacks import ConcatenateIdLogs, ModelCheckpointOnStart
from masskit_ai.spectrum.peptide.peptide_callbacks import PeptideCB
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import time
import logging

# set up matplotlib to use a non-interactive back end
try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    matplotlib = None


"""
train a spectrum prediction model
"""


def create_experiment_name(config):
    """
    create a unique experiment identifier

    :param config: configuration
    """
    if config.experiment_name == "default_name":
        config.experiment_name = (str(date.today()) + '_' + config.logging.user ) if config.logging.user else str(date.today())
        config.experiment_name +=  '_' + list(config.ml.model)[0] + "_" + config.ml.loss.loss_function


def setup_loggers(config, model, loader, artifacts):
    # log experiment parameters to mlflow and tensorboard
    if config.logging.loggers:
        loggers = {
            logger: class_for_name(config.paths.modules.loggers, logger)(
                config, model, loader, artifacts
            )
            for logger in config.logging.loggers
        }
    else:
        loggers = {}
        
    if 'output_type' in config.ml.model and config.ml.model.output_type == 'spectra':
        # graphing callback
        callbacks = [PeptideCB(config, loggers=loggers)]
    else:
        callbacks = []

    # log training record ids if requested
    if "log_ids" in config.input.train and config.input.train.log_ids:
        callbacks.append(ConcatenateIdLogs())

    # save best model callback
    if "MSMLFlowLogger" in loggers:
        checkpoint_logger = ModelCheckpointOnStart(
            config=config,
                monitor="val_loss",
                save_top_k=config.logging.save_top_k,
                filename=config.experiment_name + '_' + loggers["MSMLFlowLogger"].run_id + "_{val_loss:.4f}_{epoch:03d}",
                mode="min",
                dirpath='best_model/' + loggers["MSMLFlowLogger"].run_id
            )
        callbacks.append(checkpoint_logger)

    return loggers, callbacks


# to override the config file, use --config-name.  To override the config path, use --config-path
@hydra.main(config_path="conf", config_name="config", version_base=None)
def train_app(config: DictConfig):

    start_time = time.time()

    # turn on debugging
    set_detect_anomaly = config.setup.get('set_detect_anomaly', False)
    
    # set precision
    precision = config.ml.get('precision', 32)

    # artifact directories to log
    artifacts = {}  

    config.setup.reproducable_seed = pl.seed_everything(config.setup.reproducable_seed)
    # in future editions of pl, consider adding workers=True

    trainer = None
    create_experiment_name(config)

    # set up data loader and model
    loader = setup_datamodule(config)
    # set up lightning module
    lightning_module = class_for_name(config.paths.modules.lightning_modules,
                            config.ms.get("lightning_module", "SpectrumLightningModule"))

    if not config.ml.transfer_learning and config.input.checkpoint_in is not None:
        # resume training from checkpoint
        model = lightning_module(config)
        # update reproducable_seed to get logging to work correctly
        model.config.setup.reproducable_seed = config.setup.reproducable_seed
        
        loggers, callbacks = setup_loggers(config, model, loader, artifacts)
        
        trainer = pl.Trainer(
            accelerator=config.setup.accelerator, 
            devices=config.setup.gpus,
            num_nodes=config.setup.num_nodes,
            detect_anomaly=set_detect_anomaly,
            logger=loggers.values(),
            callbacks=callbacks,
        )
        trainer.fit(model, ckpt_path=config.input.checkpoint_in, datamodule=loader)

    else:
        if config.ml.transfer_learning:
            # transfer learning
            model = lightning_module.load_from_checkpoint(config.input.checkpoint_in)
            # optionally call a function that updates the model
            if "model_modifier" in config.ml.model and config.ml.model.model_modifier is not None:
                model_modifier = class_for_name(config.paths.modules.model_modifiers, config.ml.model.model_modifier)
                model = model_modifier(model, config)
        else:
            # normal training
            model = lightning_module(config)
            
        loggers, callbacks = setup_loggers(config, model, loader, artifacts)
        
        # train the model
        trainer = pl.Trainer(
            devices=config.setup.gpus,
            num_nodes=config.setup.num_nodes,
            logger=loggers.values(),
            max_epochs=config.ml.max_epochs,
            accelerator=config.setup.accelerator,
            callbacks=callbacks,
            limit_train_batches=config.ml.limit_train_batches,
            limit_val_batches=config.ml.get('limit_val_batches', 1.0),
            precision=precision,
            detect_anomaly=set_detect_anomaly,
            gradient_clip_val=config.ml.get('gradient_clip_val', None),
        )
        trainer.fit(model, datamodule=loader)

    # close the loggers. make sure last logger in the list is the one that saves the artifacts
    for logger in trainer.loggers:
        # save the best model as an artifact
        if isinstance(logger, MSMLFlowLogger):
            for callback in trainer.callbacks:
                if isinstance(callback, ModelCheckpoint) and callback.best_model_path:
                    logger.experiment.log_artifact(logger.run_id, callback.best_model_path)
                    logging.info(f'path to best model is {callback.best_model_path}')
                    # for use in testing.  use +create_symlink=best_model
                    if 'create_symlink' in config:
                        os.symlink(callback.best_model_path, config.create_symlink)
        logger.close(artifacts=artifacts)

    logging.info(f"elapsed wall clock time={time.time() - start_time} sec")

if __name__ == "__main__":
    filter_pytorch_lightning_warnings()
    train_app()

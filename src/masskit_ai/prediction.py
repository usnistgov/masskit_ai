import numpy as np
import torch

def apply_dropout(model):
    """
    for use by torch apply to turn on dropout in a model in eval mode.

    :param model: the model
    """
    if type(model) == torch.nn.Dropout:
        model.train()


def prep_model_for_prediction(model, dropout=False, cpu=False):
    """
    prepare the model for inference

    :param model: the model
    :param dropout: should dropout be turned on?
    :param model: place model on cpu?
    """
    if cpu:
        model.cpu()
    model.eval()  # eval mode turns off training flag in all layers of model
    if dropout:  # optionally turn dropout back on
        model.model.apply(apply_dropout)

import pytest
from pytest import approx
import os
import re
from masskit_ai.apps.ml.peptide import predict
from masskit.apps.process.libraries.batch_converter import batch_converter_app


def test_predict_peptide(config_predict_peptide):
    predict.main(config_predict_peptide)
    assert os.path.exists(config_predict_peptide.predict.output_prefix+'.'+config_predict_peptide.predict.output_suffixes[0])
    with open(config_predict_peptide.predict.output_prefix+'.'+config_predict_peptide.predict.output_suffixes[1]) as f:
        data = f.read()
        assert re.search(r'1915\.9524\t2\.46.*"y17\+i/0.0ppm"',data) is not None
        assert re.search(r'1817\.9044\t0\.11.*"y16\+i/0.0ppm"', data) is not None
        assert data.count('Name:') == 6

def test_predict_airi(config_predict_airi):
    predict.main(config_predict_airi)
    with open(config_predict_airi.predict.output_prefix+'.'+config_predict_airi.predict.output_suffixes[0]) as f:
        data = f.read()
        assert re.search(r'1087\..*,21\..*',data) is not None
        assert re.search(r'1640\..*,65\..*', data) is not None


# test should technically be in masskit unit tests, but needs a predicted file
def test_batch_converter(config_batch_converter):
    batch_converter_app(config_batch_converter)

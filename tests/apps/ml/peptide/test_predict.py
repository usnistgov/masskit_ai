import pytest
from pytest import approx
import os
import re
from masskit_ai.apps.ml.peptide import predict

def test_predict_peptide(config_predict_peptide,create_predicted_peptide_arrow_file):
    predict.main(config_predict_peptide)
    assert os.path.exists(create_predicted_peptide_arrow_file[0])
    with open(create_predicted_peptide_arrow_file[1]) as f:
        data = f.read()
        assert re.search(r'1915\.9524\t2\.46.*"y17\+i/0.0ppm"',data) is not None
        assert re.search(r'1817\.9044\t0\.11.*"y16\+i/0.0ppm"', data) is not None
        assert data.count('Name:') == 6

def test_predict_airi(config_predict_airi):
    predict.main(config_predict_airi)
#    return config_predict_airi.predict.output_prefix+"."+config_predict_airi.predict.output_suffixes[0]

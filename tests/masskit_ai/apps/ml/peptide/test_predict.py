import os
import re

from pytest import approx

from masskit_ai.apps.ml.peptide import predict


def test_predict_peptide(config_predict_peptide):
    predict.predict_app(config_predict_peptide)
    assert os.path.exists(config_predict_peptide.predict.output_prefix+'.'+config_predict_peptide.predict.output_suffixes[0])
    with open(config_predict_peptide.predict.output_prefix+'.'+config_predict_peptide.predict.output_suffixes[1]) as f:
        data = f.read()
        assert re.search(r'1915\.9524\t2\.46.*"y17\+i/0.0ppm"',data) is not None
        assert re.search(r'1817\.9044\t0\.11.*"y16\+i/0.0ppm"', data) is not None
        assert data.count('Name:') == 6

def test_predict_airi_csv(config_predict_airi_csv):
    predict.predict_app(config_predict_airi_csv)
    with open(config_predict_airi_csv.predict.output_prefix+'.'+config_predict_airi_csv.predict.output_suffixes[0]) as f:
        data = f.read()
        assert re.search(r'1530\..*,12\..*',data) is not None
        assert re.search(r'602\..*,29\..*', data) is not None
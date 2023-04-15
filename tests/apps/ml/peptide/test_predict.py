import pytest
from pytest import approx
import os
import re

def test_predict_peptide_main(create_predicted_peptide_arrow_file):
    assert os.path.exists(create_predicted_peptide_arrow_file[0])
    with open(create_predicted_peptide_arrow_file[1]) as f:
        data = f.read()
        assert re.search(r'1915\.9524\t2\.46.*"y17\+i/0.0ppm"',data) is not None
        assert re.search(r'1817\.9044\t0\.11.*"y16\+i/0.0ppm"', data) is not None
        assert data.count('Name:') == 6

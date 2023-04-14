import pytest
from pytest import approx
import os

def test_predict_peptide_main(create_predicted_peptide_parquet_file):
    assert os.path.exists(create_predicted_peptide_parquet_file[0])
    with open(create_predicted_peptide_parquet_file[1]) as f:
        data = f.read()
        assert '1915.9524	2.4601402	"y17+i/0.0ppm"' in data
        assert '1817.9044	0.11164878	"y16+i/0.0ppm"' in data
        assert data.count('Name:') == 5

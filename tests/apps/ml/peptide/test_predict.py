import pytest
from pytest import approx
import os

def test_predict_peptide_main(create_predicted_peptide_parquet_file):
    assert os.path.exists(create_predicted_peptide_parquet_file)

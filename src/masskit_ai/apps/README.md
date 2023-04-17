# Various library operations

- ml
  - peptide
    - create_peptide_library: create a library of barcode spectra that have standard ions of a fixed intensity.
      - example command line `python apps/ms/peptide/create_peptide_library.py num=2000000 output_prefix=test`
      - arguments
        - `predict.num`: number of spectra
        - `predict.min_length`: minimum length of peptides
        - `predict.max_length`: maximum length of peptides
        - `predict.min_charge`: minimum charge of peptide
        - `predict.max_charge`: maximum charge of peptide
        - `predict.min_ev`: minimum eV of peptide
        - `predict.max_ev`: maximum eV of peptide
        - `predict.mod_list`: list of modifications to use
        - `predict.output_prefix`: prefix of output filenames
        - `predict.output_suffixes`: output filename file extensions

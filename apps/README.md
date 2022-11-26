# Various library operations

- ml
  - peptide
    - create_peptide_library: create a library of barcode spectra that have standard ions of a fixed intensity.
      - example command line `python apps/ms/peptide/create_peptide_library.py num=2000000 output_prefix=test`
      - arguments
        - `num`: number of spectra
        - `min_length`: minimum length of peptides
        - `max_length`: maximum length of peptides
        - `min_charge`: minimum charge of peptide
        - `max_charge`: maximum charge of peptide
        - `min_ev`: minimum eV of peptide
        - `max_ev`: maximum eV of peptide
        - `mod_list`: list of modifications to use
        - `output_prefix`: prefix of output filenames
        - `output_suffixes`: output filename file extensions

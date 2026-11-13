# Pre-registration

- `PREREGISTRATION_external_validation.md`: the registered plan (Stage 1). Later stages are added as Amendment A and as the hashes of the prediction files, each under its own tag.
- `SHA256SUMS`: SHA-256 of every code file and training-data input locked at Stage 1, with paths relative to the project's code folder.

The code is released with the journal paper. To check it against this record, run this from the released code folder:

    shasum -a 256 -c SHA256SUMS

The three training-data inputs are derived from the COD and are not redistributed here.

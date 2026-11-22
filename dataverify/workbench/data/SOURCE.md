# Palmer Penguins data

`palmer-penguins.csv` is a byte-for-byte copy of the simplified `penguins` dataset in the palmerpenguins R package. The 344 records describe observations collected in the Palmer Archipelago, Antarctica, from 2007 to 2009. Kristen Gorman and Palmer Station LTER collected the data. Allison Horst, Alison Hill and Kristen Gorman published the R package. The maintainers release the data under CC0. [Dataset, license and original study citations](https://allisonhorst.github.io/palmerpenguins/) · [Package DOI](https://doi.org/10.5281/zenodo.3960218) · [Original study](https://doi.org/10.1371/journal.pone.0090081)

- Fixed upstream commit `8957207b78d6ccd1b4654a9dd9c9041b657478ab`. [Original CSV](https://github.com/allisonhorst/palmerpenguins/blob/8957207b78d6ccd1b4654a9dd9c9041b657478ab/inst/extdata/penguins.csv)
- SHA-256 `f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93` (15,241 bytes).
- SQLite uses all eight original columns and preserves every record. The importer converts `NA` to SQL `NULL` and numeric measurements to the corresponding integer or real types. Counts describe the sample and its measurement coverage.

Original dataset identifiers at the Environmental Data Initiative are available for [Adélie penguins](https://doi.org/10.6073/pasta/98b16d7d563f265cb52372c8ca99e60f), [Gentoo penguins](https://doi.org/10.6073/pasta/7fca67fb28d56ee2ffa3d9370ebda689) and [Chinstrap penguins](https://doi.org/10.6073/pasta/c14dfcfada8ea13a17536e73eb6fbe9e).

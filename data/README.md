# Datasets

Nothing in this folder is committed. The two datasets total roughly 3 GB and both are licensed for research use with attribution rather than redistribution.

## PlantVillage (laboratory)

```bash
pip install 'kaggle>=1.8'
kaggle auth login # browser flow; or put a token at ~/.kaggle/access_token
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p data/raw/plantvillage --unzip
```

## PlantDoc (field)

```bash
git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git data/raw/_plantdoc
mkdir -p data/raw/plantdoc
mv data/raw/_plantdoc/train data/raw/_plantdoc/test data/raw/plantdoc/
rm -rf data/raw/_plantdoc
```

Kaggle mirror if the clone is slow: `nirmalsankalana/plantdoc-dataset`.

## Citation

- Hughes, D. P. and Salathé, M. (2015) *An open access repository of images on
  plant health.* arXiv:1511.08060. CC BY-SA.
- Singh, D. et al. (2020) *PlantDoc: A dataset for visual plant disease
  detection.* CoDS-COMAD. CC BY 4.0.
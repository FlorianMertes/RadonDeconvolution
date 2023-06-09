

[![DOI](https://zenodo.org/badge/622906699.svg)](https://zenodo.org/badge/latestdoi/622906699)



# RadonDeconvolution

- contents:
  - Main Filtering and Smoothing functions and demo in toplevel Jupyter-Notebook (.ipynb)
  - this requires/imports IntegrationDiscretization.py which contains analytical solutions to the forward dynamics of the model derived from symbolic computation - see       discretization_gen.py for an example
  - Spectrometric data of open (Radon emanating source) in data/gamma-spectra/open_source and of background in data/gamma-spectra/background
  - environmental data in sqlite database in data/temprh.db
  
  
- Tested with 
  Python 3.10.5 with JAX (https://github.com/google/jax) 0.4.8 (JAX deprecations may break the code), Tensorflow Probability (Jax Substrate, 0.17) and required preqrequisites NumPy (1.23.1), SciPy (1.9.0) and Sympy (1.10.1)

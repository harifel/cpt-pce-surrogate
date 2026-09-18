# Surrogate Models for Forward Simulation and Inverse Parameter Identification Using Numerical CPT Data

Code for the paper: *Surrogate models for forward simulation and inverse parameter identification using numerical cone penetration test data.*

PCE surrogate models (built with [UQPyLab](https://www.uqlab.com/)) are trained on numerical CPT simulation data to predict CPT outputs from soil parameters (forward) and recover soil parameters from CPT measurements (inverse). Sobol sensitivity indices are extracted analytically from the PCE coefficients.

---

## Structure

```
├── data/               - simulation database (not tracked)
├── graphics/           - output figures
├── models/             - saved PCE models
├── src/
│   ├── 1000_forward_model_PCE.ipynb               - forward PCE (full dataset)
│   ├── 1010_inverse_model_PCE.ipynb               - inverse PCE
│   ├── data_loading.py / pce_utils.py / plotting.py / utils.py
│   ├── config.py           - UQPyLab token (not tracked)
│   └── config.example.py   - token template
└── environment.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows — or: source venv/bin/activate
pip install -r environment.txt
cp src/config.example.py src/config.py   # then add your UQCloud token
```

Place the simulation database in `data/` before running the notebooks.

## Dependencies

`uqpylab`, `numpy`, `scipy`, `pandas`, `openpyxl`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyterlab`

## Citation

> [to be added upon publication]

## License

See `LICENSE`.

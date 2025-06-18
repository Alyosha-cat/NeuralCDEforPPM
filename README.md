# Neural CDE for Predictive Process Monitoring (PPM)
---

This repository supports the project _"Applying Neural CDEs to Business Process Prediction: A Study on Irregular and Missing Data."_ It extends the [SuffixTransformerNetwork](https://github.com/BrechtWts/SuffixTransformerNetwork) codebase by integrating a Neural Controlled Differential Equation (Neural CDE) architecture.

The implemented framework enables evaluation against baseline models including:
- ED-LSTM
- CRTP-LSTM
- SuTraN

Experiments are conducted on two real-world event log datasets:
- **BPIC17**
- **BPIC19**

---

## 🛠 Dependencies

This project requires the following additional libraries:
- [`torchcde`](https://github.com/patrick-kidger/torchcde)
- `torchdiffeq`
- plus any packages listed in the original repository

---

## 📁 Project Structure

### 1. Data Preparation
There are no modifications to data preprocessing compared to the original SuffixTransformerNetwork repository. Please refer to the original repo for:
- dataset load,
- trace extraction,
- prefix-suffix generation.

### 2. Running Models

To run experiments, use the provided Jupyter notebooks:

```bash
Run + [Model Name] + [Dataset Code].ipynb
```

For example:
- `RunNODE17.ipynb`: Runs the Neural CDE model on BPIC17
- `RunED_LSTM19.ipynb`: Runs the ED-LSTM model on BPIC19

Each notebook includes:
- training process,

### 3. Results

To view performance metrics (DLS, MAE), please use the provided jupyter notebooks:

```bash
check.ipynb
check_others + baselinemodel.ipynb
```
For example:

- `check.ipynb`: check nueral CDE model results
- `check_others ED`: check ED-LSTM model results

---

## 🔬 Neural CDE Implementation Details

### Training and Evaluation

- Trainer script: `TRAIN_EVAL_NODE_ND.py`

### Model Definition

- Model architecture: `NODE/NODE.py`

### Missing Data Simulation

- Dropout logic for prefix sequences is implemented in: `NODE/tensor_utils.py`
- This utility simulates missing events during test time by applying a dropout mask and re-allocating values to the final time step


# DeepSynergy

DeepSynergy is a PyTorch-based framework for estimating *union information* and high-order *synergy* in multivariate systems. It provides a modular implementation of a neural estimator that computes the synergy between multiple source variables \(X_1, \dots, X_n\) and a target \(Y\), using adversarial training to enforce conditional independence constraints.

---

## 🧠 Key Features

- Support for binary, categorical, and continuous variables
- Modular encoder/decoder architecture
- Includes logic gates and Gaussian systems examples
- Easily extensible to real-world neural data

---

## 📦 Installation

Install the package and its dependencies with:

```bash
pip install -r requirements.txt
```

You can also install it locally:

```bash
pip install -e .
```

---

## 📁 Repository Structure

```text
deepsynergy/
├── __init__.py               # Public API surface
├── encoders.py               # Gaussian encoders (VAE-style)
├── decoders.py               # Binary, categorical, Gaussian, and mixture decoders
├── model.py                  # Core DeepSynergy model
├── optim.py                  # Optimizer builder (gen/disc)
├── utils_data.py             # Predefined logic gates and Gaussian datasets
├── utils_training.py         # Training loops and schedulers
examples/
├── example_decoders.ipynb    # Tests for decoder accuracy vs analytical entropies
├── example_gaussian_triplet.ipynb  # Triplet synergy estimation (Barrett-style)
├── example_logic_gates.ipynb       # Logic gate synergy (AND, XOR, etc.)
requirements.txt
pyproject.toml
LICENSE
README.md
```

---

## 🧪 Examples

Run the notebooks in the `examples/` folder to validate the framework.

```bash
jupyter notebook examples/example_decoders.ipynb
```

- `example_decoders.ipynb`: Verifies that decoders estimate conditional entropy correctly.
- `example_gaussian_triplet.ipynb`: Replicates known synergy computation in Gaussian systems.
- `example_logic_gates.ipynb`: Computes synergy on logic gates with theoretical ground-truth.

---

## 🔬 Citation

If you use this code in your research, please consider citing the corresponding publication (link TBD).

---

## 📝 License

This project is licensed under the terms of the MIT license. See the `LICENSE` file for details.

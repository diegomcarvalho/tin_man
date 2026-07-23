# Build and install

```
python3 -m venv .venv
source .venv/bin/activate
pip install maturin
cd tin_man_py
maturin develop --release
```
maturin develop compiles the Rust crate and installs it directly into your active virtual
environment as an importable module. For a distributable wheel instead, run maturin 
`build --release`, which produces a `.whl` in `target/wheels/` that you can pip install anywhere.

# Using it from Python

```
import tin_man_py
# Standard WiSARD
w = tin_man_py.Wisard(input_size=64, address_size=8, confidence_threshold=0.1,
bleaching_enabled=True, ignore_zero=False)
w.train([1, 0, 1, 0] * 16, "class_a")
w.train([0, 1, 0, 1] * 16, "class_b")
print(w.classify([1, 0, 1, 0] * 16)) # ('class_a', 0.875)
w.save_to_file("model.json", "json")
w2 = tin_man_py.Wisard.load_from_file("model.json", "json")
# ClusWiSARD
clus = tin_man_py.ClusWisard(64, 8, 0.3, 20, 5, 0.1, True, False)
clus.train([1, 1, 0, 0] * 16, "cold")
print(clus.classify([1, 1, 0, 0] * 16))
# Regression WiSARD
rew = tin_man_py.RegressionWisard(64, 8, 1)
rew.train([1, 0, 1, 0] * 16, 10.5)
print(rew.predict([1, 0, 1, 0] * 16))
```

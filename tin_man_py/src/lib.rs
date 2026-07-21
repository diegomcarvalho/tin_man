use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tin_man::{ClusWisard, FileFormat, RegressionWisard, Wisard};

fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn parse_format(format: &str) -> PyResult<FileFormat> {
    match format.to_lowercase().as_str() {
        "json" => Ok(FileFormat::Json),
        "binary" | "bin" => Ok(FileFormat::Binary),
        other => Err(PyValueError::new_err(format!(
            "unknown format '{other}', expected 'json' or 'binary'"
        ))),
    }
}

/// Python wrapper around tin_man::Wisard.
#[pyclass(name = "Wisard")]
struct PyWisard {
    inner: Wisard,
}

#[pymethods]
impl PyWisard {
    #[new]
    #[pyo3(signature = (input_size, address_size, confidence_threshold, bleaching_enabled, ignore_zero))]
    fn new(
        input_size: usize,
        address_size: usize,
        confidence_threshold: f64,
        bleaching_enabled: bool,
        ignore_zero: bool,
    ) -> Self {
        PyWisard {
            inner: Wisard::new(input_size, address_size, confidence_threshold, bleaching_enabled, ignore_zero),
        }
    }

    fn train(&mut self, input: Vec<u8>, label: &str) {
        self.inner.train(&input, label);
    }

    fn classify(&self, input: Vec<u8>) -> Option<(String, f64)> {
        self.inner.classify(&input)
    }

    fn save_to_file(&self, path: &str, format: &str) -> PyResult<()> {
        let fmt = parse_format(format)?;
        self.inner.save_to_file(path, fmt).map_err(to_py_err)
    }

    #[staticmethod]
    fn load_from_file(path: &str, format: &str) -> PyResult<Self> {
        let fmt = parse_format(format)?;
        let inner = Wisard::load_from_file(path, fmt).map_err(to_py_err)?;
        Ok(PyWisard { inner })
    }
}

/// Python wrapper around tin_man::ClusWisard.
#[pyclass(name = "ClusWisard")]
struct PyClusWisard {
    inner: ClusWisard,
}

#[pymethods]
impl PyClusWisard {
    #[new]
    #[pyo3(signature = (input_size, address_size, min_score, threshold, discriminator_limit, confidence_threshold, bleaching_enabled, ignore_zero))]
    fn new(
        input_size: usize,
        address_size: usize,
        min_score: f64,
        threshold: u32,
        discriminator_limit: usize,
        confidence_threshold: f64,
        bleaching_enabled: bool,
        ignore_zero: bool,
    ) -> Self {
        PyClusWisard {
            inner: ClusWisard::new(
                input_size,
                address_size,
                min_score,
                threshold,
                discriminator_limit,
                confidence_threshold,
                bleaching_enabled,
                ignore_zero,
            ),
        }
    }

    fn train(&mut self, input: Vec<u8>, label: &str) {
        self.inner.train(&input, label);
    }

    fn classify(&self, input: Vec<u8>) -> Option<(String, f64)> {
        self.inner.classify(&input)
    }

    fn save_to_file(&self, path: &str, format: &str) -> PyResult<()> {
        let fmt = parse_format(format)?;
        self.inner.save_to_file(path, fmt).map_err(to_py_err)
    }

    #[staticmethod]
    fn load_from_file(path: &str, format: &str) -> PyResult<Self> {
        let fmt = parse_format(format)?;
        let inner = ClusWisard::load_from_file(path, fmt).map_err(to_py_err)?;
        Ok(PyClusWisard { inner })
    }
}

/// Python wrapper around tin_man::RegressionWisard.
#[pyclass(name = "RegressionWisard")]
struct PyRegressionWisard {
    inner: RegressionWisard,
}

#[pymethods]
impl PyRegressionWisard {
    #[new]
    fn new(input_size: usize, address_size: usize, min_zero: u32) -> Self {
        PyRegressionWisard {
            inner: RegressionWisard::new(input_size, address_size, min_zero),
        }
    }

    fn train(&mut self, input: Vec<u8>, target: f64) {
        self.inner.train(&input, target);
    }

    fn predict(&self, input: Vec<u8>) -> Option<f64> {
        self.inner.predict(&input)
    }

    fn save_to_file(&self, path: &str, format: &str) -> PyResult<()> {
        let fmt = parse_format(format)?;
        self.inner.save_to_file(path, fmt).map_err(to_py_err)
    }

    #[staticmethod]
    fn load_from_file(path: &str, format: &str) -> PyResult<Self> {
        let fmt = parse_format(format)?;
        let inner = RegressionWisard::load_from_file(path, fmt).map_err(to_py_err)?;
        Ok(PyRegressionWisard { inner })
    }
}

/// Python module definition. Name must match `lib.name` in Cargo.toml.
#[pymodule]
fn tin_man_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWisard>()?;
    m.add_class::<PyClusWisard>()?;
    m.add_class::<PyRegressionWisard>()?;
    Ok(())
}
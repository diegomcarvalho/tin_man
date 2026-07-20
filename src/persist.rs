use serde::{de::DeserializeOwned, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Result as IoResult};
use std::path::Path;

/// File format used for persistence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    /// Human-readable JSON (larger, easy to inspect/diff).
    Json,
    /// Compact binary format via bincode (smaller, faster to load).
    Binary,
}

pub(crate) fn save_to_file<T: Serialize>(value: &T, path: impl AsRef<Path>, format: FileFormat) -> IoResult<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    match format {
        FileFormat::Json => {
            serde_json::to_writer_pretty(writer, value).map_err(to_io_err)?;
        }
        FileFormat::Binary => {
            bincode::serialize_into(writer, value).map_err(to_io_err)?;
        }
    }
    Ok(())
}

pub(crate) fn load_from_file<T: DeserializeOwned>(path: impl AsRef<Path>, format: FileFormat) -> IoResult<T> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let value = match format {
        FileFormat::Json => serde_json::from_reader(reader).map_err(to_io_err)?,
        FileFormat::Binary => bincode::deserialize_from(reader).map_err(to_io_err)?,
    };
    Ok(value)
}

fn to_io_err<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
}
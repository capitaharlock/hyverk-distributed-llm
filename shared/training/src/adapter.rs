// LoRA adapter serialization / deserialization + FedAvg aggregation.
// Adapters are stored as safetensors bytes for compact transport.
// FedAvg: coordinator averages adapters from N nodes → merged adapter.

use candle_core::{Result, Tensor};
use candle_nn::VarMap;
use std::collections::HashMap;

/// Serialize all LoRA vars in a VarMap to safetensors bytes.
/// Saves to a tempfile then reads back — candle 0.8 doesn't have save_to_buffer.
pub fn serialize_adapter(
    varmap: &VarMap,
) -> std::result::Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let named = varmap.data().lock().unwrap();

    // Build name→tensor map
    let tensors: HashMap<String, Tensor> = named
        .iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
        .collect();
    drop(named);

    // Save to temp file, read back as bytes
    let tmp = tempfile::NamedTempFile::new()?;
    candle_core::safetensors::save(&tensors, tmp.path())?;
    let bytes = std::fs::read(tmp.path())?;
    Ok(bytes)
}

/// Deserialize adapter bytes to a name→tensor map
pub fn deserialize_adapter(
    bytes: &[u8],
    device: &candle_core::Device,
) -> Result<HashMap<String, Tensor>> {
    // Write to tempfile for candle's load API
    let tmp = tempfile::NamedTempFile::new()
        .map_err(|e| candle_core::Error::Msg(format!("tempfile: {e}")))?;
    std::fs::write(tmp.path(), bytes)
        .map_err(|e| candle_core::Error::Msg(format!("write: {e}")))?;
    candle_core::safetensors::load(tmp.path(), device)
}

/// Federated Averaging: average N adapters into one.
/// All adapters must have identical keys and shapes.
pub fn fedavg(
    adapters: Vec<HashMap<String, Tensor>>,
) -> Result<HashMap<String, Tensor>> {
    if adapters.is_empty() {
        return Ok(HashMap::new());
    }
    if adapters.len() == 1 {
        return Ok(adapters.into_iter().next().unwrap());
    }

    let n = adapters.len() as f64;
    let keys: Vec<String> = adapters[0].keys().cloned().collect();

    let mut merged = HashMap::new();
    for key in &keys {
        let tensors: Vec<&Tensor> = adapters.iter()
            .filter_map(|a| a.get(key))
            .collect();

        if tensors.is_empty() {
            tracing::warn!("Key {key} missing from some adapters, skipping");
            continue;
        }

        // Sum all adapters for this key
        let mut sum = tensors[0].clone();
        for t in tensors.iter().skip(1) {
            sum = (sum + (*t).clone())?;
        }

        let avg = (sum / n)?;
        merged.insert(key.clone(), avg);
    }

    Ok(merged)
}

/// Convert merged adapter map to saveable safetensors bytes
pub fn save_merged_adapter(
    merged: &HashMap<String, Tensor>,
) -> std::result::Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let tmp = tempfile::NamedTempFile::new()?;
    candle_core::safetensors::save(merged, tmp.path())?;
    Ok(std::fs::read(tmp.path())?)
}

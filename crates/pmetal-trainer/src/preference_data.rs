//! Preference dataset loaders for DPO, SimPO, ORPO, and KTO.
//!
//! These functions load JSONL datasets and tokenize them into the typed pairs
//! expected by the respective trainers. Moved here from the former `easy` module
//! so that all training-related data loading lives in `pmetal-trainer`.

use std::io::BufRead;

use pmetal_data::Tokenizer;

use crate::dpo;
use crate::kto::KtoSample;
use crate::simpo;

/// Read a JSONL file into a vector of JSON objects.
pub fn read_jsonl_objects(path: &str) -> anyhow::Result<Vec<serde_json::Value>> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut rows = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        rows.push(serde_json::from_str(&line)?);
    }
    Ok(rows)
}

/// Return the first string value found for any of the given field names.
pub fn extract_first_string<'a>(value: &'a serde_json::Value, fields: &[&str]) -> Option<&'a str> {
    fields
        .iter()
        .find_map(|field| value.get(*field).and_then(serde_json::Value::as_str))
}

/// Encode a text segment, optionally appending EOS and truncating.
pub fn encode_segment(
    tokenizer: &Tokenizer,
    text: &str,
    max_len: usize,
    truncate_left: bool,
    append_eos: bool,
) -> anyhow::Result<Vec<u32>> {
    let mut ids = tokenizer.encode(text)?;
    if append_eos {
        if let Some(eos) = tokenizer.eos_token_id() {
            if ids.last().copied() != Some(eos) {
                ids.push(eos);
            }
        }
    }
    if ids.len() > max_len {
        ids = if truncate_left {
            ids[ids.len() - max_len..].to_vec()
        } else {
            ids[..max_len].to_vec()
        };
    }
    Ok(ids)
}

/// Load a DPO/ORPO preference dataset from JSONL.
///
/// Expected fields (checked in order): prompt/instruction/input,
/// chosen/accepted/preferred/chosen_response/output_chosen,
/// rejected/rejected_response/dispreferred/output_rejected.
pub fn load_dpo_dataset(
    path: &str,
    tokenizer: &Tokenizer,
    max_prompt_length: usize,
    max_completion_length: usize,
    truncate_prompt_left: bool,
) -> anyhow::Result<Vec<dpo::PreferencePair>> {
    let rows = read_jsonl_objects(path)?;
    let mut pairs = Vec::with_capacity(rows.len());
    for row in rows {
        let prompt = extract_first_string(&row, &["prompt", "instruction", "input"])
            .ok_or_else(|| anyhow::anyhow!("Preference row missing prompt"))?;
        let chosen = extract_first_string(
            &row,
            &[
                "chosen",
                "accepted",
                "preferred",
                "chosen_response",
                "output_chosen",
            ],
        )
        .ok_or_else(|| anyhow::anyhow!("Preference row missing chosen response"))?;
        let rejected = extract_first_string(
            &row,
            &[
                "rejected",
                "rejected_response",
                "dispreferred",
                "output_rejected",
            ],
        )
        .ok_or_else(|| anyhow::anyhow!("Preference row missing rejected response"))?;

        let prompt_ids =
            encode_segment(tokenizer, prompt, max_prompt_length, truncate_prompt_left, false)?;
        let chosen_ids = encode_segment(tokenizer, chosen, max_completion_length, false, true)?;
        let rejected_ids =
            encode_segment(tokenizer, rejected, max_completion_length, false, true)?;
        pairs.push(dpo::PreferencePair::new(prompt_ids, chosen_ids, rejected_ids));
    }
    Ok(pairs)
}

/// Load a SimPO preference dataset from JSONL.
///
/// Same field detection as [`load_dpo_dataset`] but prompt truncation is
/// always left-side and returns `simpo::PreferencePair`.
pub fn load_simpo_dataset(
    path: &str,
    tokenizer: &Tokenizer,
    max_prompt_length: usize,
    max_completion_length: usize,
) -> anyhow::Result<Vec<simpo::PreferencePair>> {
    let rows = read_jsonl_objects(path)?;
    let mut pairs = Vec::with_capacity(rows.len());
    for row in rows {
        let prompt = extract_first_string(&row, &["prompt", "instruction", "input"])
            .ok_or_else(|| anyhow::anyhow!("Preference row missing prompt"))?;
        let chosen = extract_first_string(
            &row,
            &[
                "chosen",
                "accepted",
                "preferred",
                "chosen_response",
                "output_chosen",
            ],
        )
        .ok_or_else(|| anyhow::anyhow!("Preference row missing chosen response"))?;
        let rejected = extract_first_string(
            &row,
            &[
                "rejected",
                "rejected_response",
                "dispreferred",
                "output_rejected",
            ],
        )
        .ok_or_else(|| anyhow::anyhow!("Preference row missing rejected response"))?;

        let prompt_ids = encode_segment(tokenizer, prompt, max_prompt_length, true, false)?;
        let chosen_ids = encode_segment(tokenizer, chosen, max_completion_length, false, true)?;
        let rejected_ids =
            encode_segment(tokenizer, rejected, max_completion_length, false, true)?;
        pairs.push(simpo::PreferencePair::new(
            prompt_ids,
            chosen_ids,
            rejected_ids,
        ));
    }
    Ok(pairs)
}

/// Load a KTO dataset from JSONL.
///
/// Expected fields: prompt/instruction/input, completion/response/output/answer,
/// label/rating/chosen (bool, number, or string).
pub fn load_kto_dataset(
    path: &str,
    tokenizer: &Tokenizer,
    max_prompt_length: usize,
    max_completion_length: usize,
    truncate_prompt_left: bool,
) -> anyhow::Result<Vec<KtoSample>> {
    let rows = read_jsonl_objects(path)?;
    let mut samples = Vec::with_capacity(rows.len());
    for row in rows {
        let prompt = extract_first_string(&row, &["prompt", "instruction", "input"])
            .ok_or_else(|| anyhow::anyhow!("KTO row missing prompt"))?;
        let response =
            extract_first_string(&row, &["completion", "response", "output", "answer"])
                .ok_or_else(|| anyhow::anyhow!("KTO row missing response"))?;
        let label = row
            .get("label")
            .or_else(|| row.get("rating"))
            .or_else(|| row.get("chosen"))
            .ok_or_else(|| anyhow::anyhow!("KTO row missing label"))?;

        let is_desirable = match label {
            serde_json::Value::Bool(value) => *value,
            serde_json::Value::Number(value) => value.as_i64().unwrap_or_default() > 0,
            serde_json::Value::String(value) => matches!(
                value.to_ascii_lowercase().as_str(),
                "desirable" | "good" | "preferred" | "true" | "yes" | "1"
            ),
            _ => false,
        };

        let prompt_ids = encode_segment(
            tokenizer,
            prompt,
            max_prompt_length,
            truncate_prompt_left,
            false,
        )?;
        let response_ids =
            encode_segment(tokenizer, response, max_completion_length, false, true)?;
        samples.push(KtoSample::new(prompt_ids, response_ids, is_desirable));
    }
    Ok(samples)
}

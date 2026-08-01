// tokenizer.rs
//
// Implements SentencePiece-style Byte-Pair Encoding (BPE), the scheme
// Llama-family models use to convert between text and token IDs. The
// vocabulary and merge rules aren't things we invent -- they were learned
// during the model's training and are embedded directly in the GGUF file's
// metadata, so this module's job is mostly "read the rules out of the file,
// then apply them correctly."
//
// Two ideas make this scheme work:
//   1. Vocabulary: a fixed list of known text fragments ("pieces"), each
//      with a unique ID. Common whole words get their own single piece;
//      rare or long words are built from several smaller pieces.
//   2. Merge rules: an ordered list of "these two adjacent pieces should be
//      combined into one" instructions. Encoding starts by treating each
//      raw character as its own piece, then repeatedly applies whichever
//      valid merge appears earliest in this list, until no more apply.
//      Earlier-listed merges represent more common patterns, so applying
//      them first produces the same segmentation the model was trained on.

use std::collections::HashMap;
use crate::gguf::{GgufFile, GgufValue};

// SentencePiece represents a space as this special character, not a literal
// ASCII space. Every token boundary in the training text was marked this
// way, so we have to do the same at encode time and undo it at decode time.
const SPACE_MARKER: char = '\u{2581}';

pub struct Tokenizer {
    token_to_id: HashMap<String, usize>,
    id_to_token: Vec<String>,
    // Maps a pair of pieces to its rank (position in the merge list).
    // Lower rank = higher priority = this merge was more common in
    // training and should be applied before later-ranked merges.
    merge_ranks: HashMap<(String, String), usize>,
    bos_token_id: usize,
    eos_token_id: usize,
}

fn get_string_array(file: &GgufFile, key: &str) -> Vec<String> {
    file.metadata.iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| if let GgufValue::Array(items) = v { Some(items) } else { None })
        .unwrap_or_else(|| panic!("missing metadata array: {key}"))
        .iter()
        .map(|item| match item {
            GgufValue::String(s) => s.clone(),
            other => panic!("expected string array element for {key}, got {other:?}"),
        })
        .collect()
}

fn get_u32(file: &GgufFile, key: &str) -> u32 {
    file.metadata.iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| if let GgufValue::U32(n) = v { Some(*n) } else { None })
        .unwrap_or_else(|| panic!("missing metadata key: {key}"))
}

pub fn load_tokenizer(file: &GgufFile) -> Tokenizer {
    let id_to_token = get_string_array(file, "tokenizer.ggml.tokens");
    let merges_raw = get_string_array(file, "tokenizer.ggml.merges");

    let token_to_id: HashMap<String, usize> = id_to_token.iter()
        .enumerate()
        .map(|(id, tok)| (tok.clone(), id))
        .collect();

    // Each merge entry is stored as "left right" (space-separated), meaning
    // "combine piece `left` with piece `right`". The entry's position in
    // this list is its priority.
    let merge_ranks: HashMap<(String, String), usize> = merges_raw.iter()
        .enumerate()
        .map(|(rank, entry)| {
            let mut parts = entry.splitn(2, ' ');
            let left = parts.next().expect("malformed merge entry").to_string();
            let right = parts.next().expect("malformed merge entry").to_string();
            ((left, right), rank)
        })
        .collect();

    let bos_token_id = get_u32(file, "tokenizer.ggml.bos_token_id") as usize;
    let eos_token_id = get_u32(file, "tokenizer.ggml.eos_token_id") as usize;

    Tokenizer { token_to_id, id_to_token, merge_ranks, bos_token_id, eos_token_id }
}

// A byte that doesn't correspond to any whole vocabulary piece (rare
// symbols, emoji, etc.) falls back to a special per-byte token, formatted
// like "<0x0A>" for a newline. This guarantees every possible input string
// can be encoded, even characters the vocabulary never saw directly.
fn byte_fallback_token(byte: u8) -> String {
    format!("<0x{byte:02X}>")
}

impl Tokenizer {
    // Encoding proceeds in three stages:
    //   1. Normalize: replace spaces with the SentencePiece space marker,
    //      and prepend one, since SentencePiece treats the very start of
    //      text as if it were preceded by a space.
    //   2. Seed: split into individual Unicode characters as the starting
    //      set of pieces, falling back to per-byte tokens for anything not
    //      in the vocabulary as a single character.
    //   3. Merge: repeatedly find the adjacent pair of pieces with the
    //      lowest merge rank (highest priority) and combine them, until no
    //      adjacent pair has a known merge rule left.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let normalized: String = text.replace(' ', &SPACE_MARKER.to_string());
        let normalized = format!("{SPACE_MARKER}{normalized}");

        let mut pieces: Vec<String> = normalized.chars()
            .map(|c| {
                let s = c.to_string();
                if self.token_to_id.contains_key(&s) {
                    s
                } else {
                    // Character isn't a known single-piece token; fall back
                    // to encoding each of its UTF-8 bytes individually.
                    let mut buf = [0u8; 4];
                    let bytes = c.encode_utf8(&mut buf).as_bytes();
                    bytes.iter().map(|b| byte_fallback_token(*b)).collect::<Vec<_>>().join("\u{0}")
                }
            })
            .flat_map(|s| s.split('\u{0}').map(String::from).collect::<Vec<_>>())
            .collect();

        loop {
            // Scan every adjacent pair, find the one with the lowest merge
            // rank (i.e. the merge that was most common in training data).
            let mut best_rank: Option<usize> = None;
            let mut best_index: Option<usize> = None;

            for i in 0..pieces.len().saturating_sub(1) {
                let pair = (pieces[i].clone(), pieces[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if best_rank.is_none_or(|best| rank < best) {
                        best_rank = Some(rank);
                        best_index = Some(i);
                    }
                }
            }

            match best_index {
                Some(i) => {
                    let merged = format!("{}{}", pieces[i], pieces[i + 1]);
                    pieces.splice(i..=i + 1, [merged]);
                }
                None => break, // no known merge applies anywhere -- done
            }
        }

        pieces.iter()
            .map(|p| {
                *self.token_to_id.get(p)
                    .unwrap_or_else(|| panic!("piece not found in vocabulary after merging: {p:?}"))
            })
            .collect()
    }

    // Decoding is the reverse: look up each ID's text piece, concatenate
    // them, then turn the space marker back into literal spaces. Byte
    // fallback tokens are converted back into their raw byte, and adjacent
    // fallback bytes are recombined into whatever UTF-8 character they
    // originally represented.
    pub fn decode(&self, token_ids: &[usize]) -> String {
        let mut raw_bytes: Vec<u8> = Vec::new();

        for &id in token_ids {
            // Skip special tokens (BOS/EOS) in the human-readable output --
            // they're control signals for the model, not part of the text.
            if id == self.bos_token_id || id == self.eos_token_id {
                continue;
            }

            let piece = self.id_to_token.get(id)
                .unwrap_or_else(|| panic!("token id out of range: {id}"));

            if piece.starts_with("<0x") && piece.ends_with('>') {
                let hex = &piece[3..piece.len() - 1];
                let byte = u8::from_str_radix(hex, 16)
                    .unwrap_or_else(|_| panic!("malformed byte-fallback token: {piece}"));
                raw_bytes.push(byte);
            } else {
                raw_bytes.extend(piece.as_bytes());
            }
        }

        let text = String::from_utf8_lossy(&raw_bytes).into_owned();
        text.replace(SPACE_MARKER, " ").trim_start().to_string()
    }

    // Decodes a single token to its raw display text, without the
    // whole-string trimming that decode() applies. Used when streaming
    // tokens one at a time, where leading spaces must be preserved so words
    // don't run together as pieces arrive incrementally -- decode() trims
    // leading whitespace because it's designed for a complete string, which
    // is the wrong behavior for a single fragment of an in-progress stream.
    pub fn decode_piece(&self, token_id: usize) -> String {
        if token_id == self.bos_token_id || token_id == self.eos_token_id {
            return String::new();
        }

        let piece = self.id_to_token.get(token_id)
            .unwrap_or_else(|| panic!("token id out of range: {token_id}"));

        if piece.starts_with("<0x") && piece.ends_with('>') {
            let hex = &piece[3..piece.len() - 1];
            let byte = u8::from_str_radix(hex, 16)
                .unwrap_or_else(|_| panic!("malformed byte-fallback token: {piece}"));
            String::from_utf8_lossy(&[byte]).into_owned()
        } else {
            piece.replace(SPACE_MARKER, " ")
        }
    }

    pub fn bos_id(&self) -> usize {
        self.bos_token_id
    }

    pub fn eos_id(&self) -> usize {
        self.eos_token_id
    }
}
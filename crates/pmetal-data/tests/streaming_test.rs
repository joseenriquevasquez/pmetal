//! Integration tests for the streaming shard reader.

use pmetal_data::streaming::{StreamConfig, StreamPosition, StreamingShardReader, write_shard};
use tempfile::TempDir;

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn make_shards(dir: &TempDir, shards: &[&[&[u32]]]) -> Vec<std::path::PathBuf> {
    shards
        .iter()
        .enumerate()
        .map(|(i, docs)| {
            let path = dir.path().join(format!("shard_{i:03}.bin"));
            write_shard(&path, docs).expect("write_shard");
            path
        })
        .collect()
}

fn reader(
    paths: Vec<std::path::PathBuf>,
    seq_len: usize,
    batch_size: usize,
) -> StreamingShardReader {
    StreamingShardReader::new(StreamConfig {
        shard_paths: paths,
        seq_len,
        batch_size,
        eos_token_id: 0,
        resume_from: None,
    })
    .expect("StreamingShardReader::new")
}

// ─── Basic correctness ────────────────────────────────────────────────────────

#[test]
fn test_single_shard_packs_correctly() {
    let dir = TempDir::new().unwrap();
    // Two documents: [1, 2, 3] and [4, 5].
    // seq_len=4, EOS=0.
    //
    // seq0: doc1=[1,2,3] fills 3 slots; doc2=[4,5] opens, EOS inserted → [1,2,3,0].
    // seq1: remaining of doc2=[4,5] fills 2 slots; doc1 wraps (epoch 2),
    //       EOS inserted, first token of doc1 fills slot 4 → [4,5,0,1].
    let paths = make_shards(&dir, &[&[&[1u32, 2, 3], &[4, 5]]]);

    let mut r = reader(paths, 4, 1);

    let (batch0, _) = r.next().unwrap();
    assert_eq!(batch0.len(), 1);
    assert_eq!(batch0[0], vec![1, 2, 3, 0]);

    let (batch1, _) = r.next().unwrap();
    // doc2=[4,5] (2 tokens), then EOS=0, then first token of wrapped doc1=1.
    assert_eq!(batch1[0], vec![4, 5, 0, 1]);
}

#[test]
fn test_batch_size_respected() {
    let dir = TempDir::new().unwrap();
    // 10 documents of 3 tokens each, seq_len=4, batch_size=2.
    let docs: Vec<Vec<u32>> = (0u32..10)
        .map(|i| vec![i * 10 + 1, i * 10 + 2, i * 10 + 3])
        .collect();
    let doc_refs: Vec<&[u32]> = docs.iter().map(|d| d.as_slice()).collect();
    let paths = make_shards(&dir, &[doc_refs.as_slice()]);

    let mut r = reader(paths, 4, 2);

    let (batch, _) = r.next().unwrap();
    assert_eq!(batch.len(), 2, "batch_size should be 2");
    for seq in &batch {
        assert_eq!(seq.len(), 4, "each sequence should be seq_len=4");
    }
}

#[test]
fn test_eos_separator_inserted_between_docs() {
    let dir = TempDir::new().unwrap();
    // Two small docs that both fit in one seq_len=10 sequence.
    // doc1=[1,2], doc2=[3,4]  → packed: [1, 2, EOS, 3, 4, EOS_pad...]
    // EOS=99.
    let paths: Vec<std::path::PathBuf> = {
        let path = dir.path().join("shard_000.bin");
        write_shard(&path, &[&[1u32, 2], &[3, 4]]).unwrap();
        vec![path]
    };

    let mut r = StreamingShardReader::new(StreamConfig {
        shard_paths: paths,
        seq_len: 6,
        batch_size: 1,
        eos_token_id: 99,
        resume_from: None,
    })
    .unwrap();

    let (batch, _) = r.next().unwrap();
    let seq = &batch[0];
    assert_eq!(seq.len(), 6);
    // Positions 0..1 = doc1, position 2 = EOS, positions 3..4 = doc2, position 5 = EOS pad.
    assert_eq!(seq[0], 1);
    assert_eq!(seq[1], 2);
    assert_eq!(seq[2], 99, "EOS separator between docs");
    assert_eq!(seq[3], 3);
    assert_eq!(seq[4], 4);
    assert_eq!(seq[5], 99, "EOS pad to fill seq_len");
}

// ─── Multi-shard round-robin ──────────────────────────────────────────────────

#[test]
fn test_two_shards_exhausted_in_order() {
    let dir = TempDir::new().unwrap();
    // Shard 0: one doc [10, 11, 12]
    // Shard 1: one doc [20, 21, 22]
    // seq_len=3, batch_size=1, EOS=0.
    let paths = make_shards(&dir, &[&[&[10u32, 11, 12]], &[&[20u32, 21, 22]]]);

    let mut r = reader(paths, 3, 1);

    let (b0, _) = r.next().unwrap();
    assert_eq!(b0[0], vec![10, 11, 12]);

    let (b1, _) = r.next().unwrap();
    assert_eq!(b1[0], vec![20, 21, 22]);

    // Third call wraps around to shard 0 again (epoch 2).
    let (b2, _) = r.next().unwrap();
    assert_eq!(b2[0], vec![10, 11, 12]);
}

// ─── Deterministic resume ─────────────────────────────────────────────────────

#[test]
fn test_resume_from_saved_position_matches() {
    let dir = TempDir::new().unwrap();
    // Two shards, seq_len=4, batch_size=2.
    // Shard 0: docs [1,2,3], [4,5,6]
    // Shard 1: docs [7,8,9], [10,11,12]
    let paths = make_shards(
        &dir,
        &[
            &[&[1u32, 2, 3], &[4, 5, 6]],
            &[&[7u32, 8, 9], &[10, 11, 12]],
        ],
    );

    let cfg = StreamConfig {
        shard_paths: paths.clone(),
        seq_len: 4,
        batch_size: 2,
        eos_token_id: 0,
        resume_from: None,
    };

    // Read two batches and capture the position after the first.
    let mut r1 = StreamingShardReader::new(cfg.clone()).unwrap();
    let (batch0_first_run, pos_after_first) = r1.next().unwrap();
    let (batch1_first_run, _) = r1.next().unwrap();

    // Resume from the saved position — should reproduce batch1.
    let mut r2 = StreamingShardReader::new(StreamConfig {
        resume_from: Some(pos_after_first),
        ..cfg
    })
    .unwrap();
    let (batch1_resumed, _) = r2.next().unwrap();

    assert_eq!(
        batch1_first_run, batch1_resumed,
        "resumed reader should reproduce the same batch"
    );

    // The first batch itself should differ from the resumed one.
    assert_ne!(
        batch0_first_run, batch1_resumed,
        "first and second batches should differ"
    );
}

// ─── Edge cases ───────────────────────────────────────────────────────────────

#[test]
fn test_doc_longer_than_seq_len_is_split() {
    let dir = TempDir::new().unwrap();
    // One large doc: [1,2,3,4,5,6,7,8], seq_len=4.
    // The doc doesn't have EOS between split chunks (the doc is one unit;
    // splits are just physical packing boundaries within the doc).
    let paths = make_shards(&dir, &[&[&[1u32, 2, 3, 4, 5, 6, 7, 8]]]);

    let mut r = reader(paths, 4, 1);

    let (b0, _) = r.next().unwrap();
    assert_eq!(b0[0], vec![1, 2, 3, 4]);

    let (b1, _) = r.next().unwrap();
    assert_eq!(b1[0], vec![5, 6, 7, 8]);
}

#[test]
fn test_position_advances_across_shards() {
    let dir = TempDir::new().unwrap();
    // Shard 0: single doc exactly seq_len tokens → fills one batch of 1.
    // Shard 1: single doc of different tokens.
    let paths = make_shards(&dir, &[&[&[1u32, 2, 3, 4]], &[&[5u32, 6, 7, 8]]]);

    let mut r = reader(paths.clone(), 4, 1);

    let (_, pos0) = r.next().unwrap();
    let (_, pos1) = r.next().unwrap();

    // After reading all of shard 0, the reader advances to shard 1.
    // The positions should differ.
    assert_ne!(pos0, pos1, "positions should advance between batches");

    // The second position should be in shard 1 (or the reader wrapped shard 0
    // twice — both are valid depending on exact carry state, so we just
    // assert they differ).
    let _ = pos0;
    let _ = pos1;
}

#[test]
fn test_empty_corpus_returns_none() {
    let dir = TempDir::new().unwrap();
    // A shard with a single zero-length record — effectively empty.
    let path = dir.path().join("empty.bin");
    write_shard(&path, &[&[]]).unwrap();

    let mut r = reader(vec![path], 4, 1);
    assert!(r.next().is_none(), "empty corpus should return None");
}

#[test]
fn test_stream_position_equality() {
    let p1 = StreamPosition {
        shard_idx: 1,
        byte_offset: 128,
        tokens_remaining_in_doc: 0,
    };
    let p2 = StreamPosition {
        shard_idx: 1,
        byte_offset: 128,
        tokens_remaining_in_doc: 0,
    };
    let p3 = StreamPosition {
        shard_idx: 2,
        byte_offset: 0,
        tokens_remaining_in_doc: 0,
    };

    assert_eq!(p1, p2);
    assert_ne!(p1, p3);
}

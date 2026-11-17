use super::*;

#[test]
fn python_positions_and_insertion_for_all_24_prompts() -> Result<()> {
    for model in [
        "Qwen3.8-27B-BF16",
        "Qwen3.6-35B-A3B-4bit",
        "THX-02",
        "d3-flash",
    ] {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/vision")
            .join(model);
        let data: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("mrope-inputs.json")).unwrap())
                .unwrap();
        for case in data["cases"].as_array().unwrap() {
            let tokens: Vec<usize> = serde_json::from_value(case["input_ids"].clone()).unwrap();
            let grids: Vec<[usize; 3]> =
                serde_json::from_value(case["image_grid_thw"].clone()).unwrap();
            let spans = case["image_spans"].as_array().unwrap();
            let embeddings: Vec<Tensor> = spans
                .iter()
                .map(|span| {
                    let bytes =
                        std::fs::read(root.join(span["embedding_file"].as_str().unwrap())).unwrap();
                    let values = bytes
                        .chunks_exact(4)
                        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                        .collect::<Vec<_>>();
                    let rows = span["length"].as_u64().unwrap() as usize;
                    Tensor::from_vec(vec![rows, values.len() / rows], values).unwrap()
                })
                .collect();
            let images: Vec<_> = spans
                .iter()
                .zip(&grids)
                .zip(&embeddings)
                .map(|((span, &grid), embedding)| SuppliedImage {
                    start: span["start"].as_u64().unwrap() as usize,
                    grid,
                    embeddings: embedding,
                    content_hash: [1; 32],
                    parameters: b"fixture",
                })
                .collect();
            let width = embeddings[0].shape()[1];
            let text =
                Tensor::from_vec(vec![tokens.len(), width], vec![-99.0; tokens.len() * width])?;
            let pad = tokens[images[0].start];
            let (output, positions) = insert_embeddings(&tokens, pad, &text, &images)?;
            let expected: Vec<Vec<Vec<usize>>> =
                serde_json::from_value(case["position_ids"].clone()).unwrap();
            for (row, actual) in positions.rows.iter().enumerate() {
                assert_eq!(
                    *actual,
                    [
                        expected[0][0][row],
                        expected[1][0][row],
                        expected[2][0][row]
                    ],
                    "{model} row {row}"
                );
                if !positions
                    .media
                    .iter()
                    .any(|m| m.start <= row && row < m.end)
                {
                    assert_eq!(output.row_slice(row)?, text.row_slice(row)?);
                }
            }
            assert_eq!(
                positions.delta_at(tokens.len())?,
                case["rope_delta"].as_i64().unwrap()
            );
            for (image, span) in images.iter().zip(&positions.media) {
                assert_eq!(
                    &output.data()[span.start * width..span.end * width],
                    image.embeddings.data()
                );
                assert!(positions.delta_at(span.start).is_ok());
                assert!(positions.delta_at(span.start + 1).is_err());
                assert!(positions.delta_at(span.end).is_ok());
            }
        }
    }
    Ok(())
}

#[test]
fn rejects_mismatched_placeholders_shapes_grids_and_nonfinite() -> Result<()> {
    let text = Tensor::from_vec(vec![6, 2], vec![0.0; 12])?;
    let embeddings = Tensor::from_vec(vec![4, 2], vec![1.0; 8])?;
    let mut image = SuppliedImage {
        start: 1,
        grid: [1, 4, 4],
        embeddings: &embeddings,
        content_hash: [0; 32],
        parameters: b"",
    };
    assert!(insert_embeddings(&[0, 9, 9, 9, 9, 0], 9, &text, &[image]).is_ok());
    image = SuppliedImage {
        start: 1,
        grid: [1, 4, 4],
        embeddings: &embeddings,
        content_hash: [0; 32],
        parameters: b"",
    };
    assert!(insert_embeddings(&[9, 9, 9, 9, 9, 0], 9, &text, &[image]).is_err());
    for grid in [
        [1, 3, 4],
        [2, 4, 4],
        [1, 0, 4],
        [1, usize::MAX - 1, usize::MAX - 1],
    ] {
        let image = SuppliedImage {
            start: 1,
            grid,
            embeddings: &embeddings,
            content_hash: [0; 32],
            parameters: b"",
        };
        assert!(insert_embeddings(&[0, 9, 9, 9, 9, 0], 9, &text, &[image]).is_err());
    }
    let bad = Tensor::from_vec(vec![4, 2], vec![f32::NAN; 8])?;
    let image = SuppliedImage {
        start: 1,
        grid: [1, 4, 4],
        embeddings: &bad,
        content_hash: [0; 32],
        parameters: b"",
    };
    assert!(insert_embeddings(&[0, 9, 9, 9, 9, 0], 9, &text, &[image]).is_err());
    Ok(())
}

#[test]
fn media_identity_covers_content_parameters_grid_and_supplied_values() -> Result<()> {
    let text = Tensor::from_vec(vec![6, 2], vec![0.0; 12])?;
    let embedding = Tensor::from_vec(vec![4, 2], vec![1.0; 8])?;
    let changed_embedding = Tensor::from_vec(vec![4, 2], vec![2.0; 8])?;
    let hash = |content, parameters: &[u8], grid, embeddings: &Tensor| -> Result<[u8; 32]> {
        let image = SuppliedImage {
            start: 1,
            grid,
            embeddings,
            content_hash: content,
            parameters,
        };
        let (_, positions) = insert_embeddings(&[0, 9, 9, 9, 9, 0], 9, &text, &[image])?;
        assert!(positions.prefix(1)?.is_none());
        assert!(positions.prefix(2).is_err());
        assert_eq!(positions.prefix(5)?.unwrap().rows.len(), 5);
        Ok(positions.media[0].hash)
    };
    let expected = hash([1; 32], b"weights-v1;resize-v1", [1, 4, 4], &embedding)?;
    assert_eq!(
        expected,
        hash([1; 32], b"weights-v1;resize-v1", [1, 4, 4], &embedding)?
    );
    assert_ne!(
        expected,
        hash([2; 32], b"weights-v1;resize-v1", [1, 4, 4], &embedding)?
    );
    assert_ne!(
        expected,
        hash([1; 32], b"weights-v2;resize-v1", [1, 4, 4], &embedding)?
    );
    assert_ne!(
        expected,
        hash([1; 32], b"weights-v1;resize-v1", [1, 2, 8], &embedding)?
    );
    assert_ne!(
        expected,
        hash(
            [1; 32],
            b"weights-v1;resize-v1",
            [1, 4, 4],
            &changed_embedding
        )?
    );
    Ok(())
}

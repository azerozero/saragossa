use super::*;

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

#[test]
fn row_i32_returns_codebook_row_by_index() -> Result<()> {
    // Codebook synthétique 2 lignes x 3 colonnes, déterministe.
    let codebook = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    assert_eq!(row_i32(&codebook, 0)?, &[1.0, 2.0, 3.0]);
    assert_eq!(row_i32(&codebook, 1)?, &[4.0, 5.0, 6.0]);
    Ok(())
}

#[test]
fn row_i32_rejects_negative_index() -> Result<()> {
    let codebook = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    assert!(row_i32(&codebook, -1).is_err());
    Ok(())
}

#[test]
fn row_i32_rejects_out_of_bounds_index() -> Result<()> {
    let codebook = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    assert!(row_i32(&codebook, 2).is_err());
    Ok(())
}

#[test]
fn add_row_in_place_accumulates_two_rvq_stages() -> Result<()> {
    // Mirroir de la boucle `rvq_decode_one` : chaque étage RVQ ajoute son
    // code dé-quantifié (dequant) dans la trame courante.
    let mut acc = Nlc::zeros(1, 3);
    acc.add_row_in_place(0, &[1.0, 2.0, 3.0])?;
    acc.add_row_in_place(0, &[0.5, -1.0, 2.0])?;
    assert_eq!(acc.row(0), &[1.5, 1.0, 5.0]);
    Ok(())
}

#[test]
fn add_row_in_place_rejects_row_length_mismatch() {
    let mut acc = Nlc::zeros(1, 3);
    assert!(acc.add_row_in_place(0, &[1.0, 2.0]).is_err());
}

#[test]
fn add_row_in_place_two_stage_reconstruction_reduces_error_each_stage() -> Result<()> {
    // Cible que le décodeur RVQ approxime par somme résiduelle sur 2 étages.
    // Les codes sont choisis à la main (comme le ferait la recherche
    // nearest-code côté encodeur) pour approximer la cible de plus en plus
    // finement : la distance à la cible doit décroître à chaque étage.
    let target = [1.0_f32, 2.0, 2.1];
    let stage1_code = [1.5_f32, 2.5, 2.5]; // approximation grossière (étage 1)
    let stage2_code = [-0.5_f32, -0.5, -0.4]; // corrige le résidu de l'étage 1

    let mut acc = Nlc::zeros(1, 3);
    let dist_before_any_stage = l2_distance(&target, acc.row(0));

    acc.add_row_in_place(0, &stage1_code)?;
    let dist_after_stage1 = l2_distance(&target, acc.row(0));

    acc.add_row_in_place(0, &stage2_code)?;
    let dist_after_stage2 = l2_distance(&target, acc.row(0));

    assert!(dist_after_stage1 < dist_before_any_stage);
    assert!(dist_after_stage2 < dist_after_stage1);
    // Les codes ont été choisis pour reconstruire exactement la cible.
    assert!(dist_after_stage2 < 1.0e-5);
    Ok(())
}

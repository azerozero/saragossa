//! Fixture DFN3 récurrente, sérialisée hors du dépôt et chargée par le service réel.
use super::*;
#[path = "../../../tests/support/dfn3_weights.rs"]
mod fixture;
use std::io::BufRead;
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

pub fn model() -> Arc<DeepFilterNet3> {
    static MODEL: std::sync::OnceLock<Arc<DeepFilterNet3>> = std::sync::OnceLock::new();
    Arc::clone(MODEL.get_or_init(|| {
        // Banque libDF ; le chargeur de production contrôle ces valeurs.
        let widths = [
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20, 24, 28,
            31, 37, 42, 50, 56, 67,
        ];
        let weights = fixture::weights(true, &widths);
        let data: Vec<_> = weights
            .iter()
            .map(|(n, (s, v))| {
                (
                    n,
                    s,
                    v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
                )
            })
            .collect();
        let views: Vec<_> = data
            .iter()
            .map(|(n, s, b)| {
                (
                    n.as_str(),
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, (*s).clone(), b)
                        .unwrap(),
                )
            })
            .collect();
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), safetensors::serialize(views, None).unwrap()).unwrap();
        Arc::new(DeepFilterNet3::load(file.path()).unwrap())
    }))
}
pub fn service() -> EnhanceService {
    let service = EnhanceService::new(None);
    *service.model.lock().unwrap() = Some(model());
    service
}
pub fn pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    client.configure(Duration::from_secs(15)).unwrap();
    server.configure(Duration::from_secs(15)).unwrap();
    (client, server)
}
pub fn headers(reader: &mut impl BufRead) -> String {
    let mut result = String::new();
    loop {
        let mut line = String::new();
        assert!(reader.read_line(&mut line).unwrap() > 0);
        result.push_str(&line);
        if line == "\r\n" {
            return result;
        }
    }
}
pub fn signal(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 7919 % 997) as f32 - 498.0) / 997.0)
        .collect()
}
pub fn oracle(input: &[f32], config: pcm::Config) -> Vec<u8> {
    let input = saragossa::denoise::resample(input, config.rate, SAMPLE_RATE).unwrap();
    let model = model();
    let mut raw = model.stream();
    let mut padded = input.clone();
    padded.resize((input.len() + 960).div_ceil(480) * 480, 0.0);
    let mut output = Vec::new();
    for chunk in padded.chunks(480) {
        let mut hop = [0.0; 480];
        raw.process_hop(chunk, &mut hop).unwrap();
        output.extend(hop);
    }
    output.extend(raw.finish());
    let output = &output[1440..1440 + input.len()];
    if config.float {
        output.iter().flat_map(|s| s.to_le_bytes()).collect()
    } else {
        output
            .iter()
            .flat_map(|s| super::super::audio::pcm_i16(*s).to_le_bytes())
            .collect()
    }
}

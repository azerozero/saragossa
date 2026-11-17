//! Identité QUIC éphémère ECDSA P-256 et empreinte publique pour le navigateur.
use super::async_io::error;
use super::*;
use sha2::{Digest, Sha256};
use std::net::SocketAddr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::serve) struct CertificateInfo {
    pub url: String,
    pub server_certificate_hashes: Vec<CertificateHash>,
    pub expires_unix: u64,
}
#[derive(Clone, serde::Serialize)]
pub(in crate::serve) struct CertificateHash {
    pub algorithm: &'static str,
    pub value: Vec<u8>,
}
pub(super) struct Identity {
    pub config: quinn::ServerConfig,
    pub info: CertificateInfo,
    #[cfg(test)]
    pub der: rustls::pki_types::CertificateDer<'static>,
}
pub(super) fn identity(addr: SocketAddr) -> ServeResult<Identity> {
    let now = SystemTime::now();
    let before = now
        .checked_sub(Duration::from_secs(60))
        .ok_or_else(|| error("horloge invalide"))?;
    let after = now
        .checked_add(Duration::from_secs(13 * 86400))
        .ok_or_else(|| error("horloge invalide"))?;
    let mut params = rcgen::CertificateParams::new(vec![
        "localhost".into(),
        "127.0.0.1".into(),
        "::1".into(),
        addr.ip().to_string(),
    ])
    .map_err(error)?;
    params.not_before = before.into();
    params.not_after = after.into();
    let key = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).map_err(error)?;
    let cert = params.self_signed(&key).map_err(error)?;
    let der = cert.der().clone();
    let key = rustls::pki_types::PrivatePkcs8KeyDer::from(key.serialize_der());
    let mut tls = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_protocol_versions(&[&rustls::version::TLS13])
    .map_err(error)?
    .with_no_client_auth()
    .with_single_cert(vec![der.clone()], key.into())
    .map_err(error)?;
    tls.alpn_protocols = vec![b"h3".to_vec()];
    let crypto = quinn::crypto::rustls::QuicServerConfig::try_from(tls).map_err(error)?;
    let mut config = quinn::ServerConfig::with_crypto(Arc::new(crypto));
    let mut transport = quinn::TransportConfig::default();
    transport
        .max_concurrent_bidi_streams(2_u32.into())
        .max_concurrent_uni_streams(3_u32.into())
        .stream_receive_window(16384_u32.into())
        .receive_window(65536_u32.into())
        .send_window(65536)
        .max_idle_timeout(Some(enhance_idle_timeout().try_into().map_err(error)?));
    config.transport_config(Arc::new(transport));
    let host = if addr.ip().is_unspecified() {
        "127.0.0.1".to_owned()
    } else if addr.is_ipv6() {
        format!("[{}]", addr.ip())
    } else {
        addr.ip().to_string()
    };
    let info = CertificateInfo {
        url: format!("https://{host}:{}/v1/audio/enhance/wt", addr.port()),
        server_certificate_hashes: vec![CertificateHash {
            algorithm: "sha-256",
            value: Sha256::digest(&der).to_vec(),
        }],
        expires_unix: after.duration_since(UNIX_EPOCH).map_err(error)?.as_secs(),
    };
    Ok(Identity {
        config,
        info,
        #[cfg(test)]
        der,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn certificate_der_has_short_lifetime_and_ecdsa_p256_key() {
        let identity = identity("127.0.0.1:0".parse().unwrap()).unwrap();
        let (_, cert) = x509_parser::parse_x509_certificate(&identity.der).unwrap();
        let validity = cert.validity();
        assert!(validity.is_valid());
        assert!(validity.not_after.timestamp() - validity.not_before.timestamp() <= 14 * 86400);
        assert_eq!(
            validity.not_after.timestamp() as u64,
            identity.info.expires_unix
        );
        assert_eq!(
            cert.public_key().algorithm.algorithm.to_id_string(),
            "1.2.840.10045.2.1"
        );
        assert_eq!(
            cert.public_key()
                .algorithm
                .parameters
                .as_ref()
                .unwrap()
                .as_oid()
                .unwrap()
                .to_id_string(),
            "1.2.840.10045.3.1.7"
        );
        assert_eq!(
            cert.signature_algorithm.algorithm.to_id_string(),
            "1.2.840.10045.4.3.2"
        );
    }
}

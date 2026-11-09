use super::*;
use std::cell::Cell;
use std::io;
use std::rc::Rc;

#[derive(Default)]
struct Writer {
    bytes: Vec<u8>,
    flushes: Rc<Cell<usize>>,
    fail_after: Option<usize>,
    fail_flush: Option<usize>,
}

impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let available = self
            .fail_after
            .unwrap_or(usize::MAX)
            .saturating_sub(self.bytes.len());
        if available == 0 {
            return Err(io::ErrorKind::BrokenPipe.into());
        }
        let count = available.min(bytes.len());
        self.bytes.extend_from_slice(&bytes[..count]);
        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        let count = self.flushes.get() + 1;
        self.flushes.set(count);
        if self.fail_flush == Some(count) {
            return Err(io::ErrorKind::TimedOut.into());
        }
        Ok(())
    }
}

fn output(samples: Vec<f32>, reached: bool) -> TtsSynthesisOutput {
    TtsSynthesisOutput {
        codes: Vec::new(),
        samples,
        sample_rate: 24_000,
        frame_limit_reached: reached,
    }
}

#[test]
fn pcm_chunks_are_s16le_with_hex_lengths_and_no_empty_chunk() -> io::Result<()> {
    let mut writer = Writer::default();
    write_pcm_chunk(&mut writer, &[])?;
    assert!(writer.bytes.is_empty());
    assert_eq!(writer.flushes.get(), 0);
    write_pcm_chunk(&mut writer, &[0.0, 1.0, -1.0, 2.0, -2.0, 0.5])?;
    assert_eq!(
        writer.bytes,
        b"c\r\n\x00\x00\xff\x7f\x01\x80\xff\x7f\x01\x80\xff\x3f\r\n"
    );
    assert_eq!(writer.flushes.get(), 1);
    Ok(())
}

#[test]
fn speech_stream_is_flushed_during_synthesis_and_matches_batch_wav() -> ServeResult<()> {
    for reached in [false, true] {
        for samples in [vec![], vec![0.0, 0.25, -0.25, 1.0, -1.0, 2.0, -2.0]] {
            let mut writer = Writer::default();
            let flushes = Rc::clone(&writer.flushes);
            handle_speech_with(
                &mut writer,
                br#"{"input":" Bonjour. ","stream":true}"#,
                Some("12"),
                |stream, text, max_frames, streaming| {
                    assert_eq!(text, "Bonjour.");
                    assert_eq!(max_frames, 12);
                    assert!(streaming);
                    send_speech_stream(stream, 24_000, |on_audio| {
                        assert_eq!(flushes.get(), 1);
                        on_audio(&[])?;
                        for (index, chunk) in samples.chunks(2).enumerate() {
                            on_audio(chunk)?;
                            assert_eq!(flushes.get(), index + 2);
                        }
                        Ok(output(samples.clone(), reached))
                    })
                },
            )?;
            let split =
                find_sub(&writer.bytes, b"\r\n\r\n").expect("invariant: en-têtes écrits") + 4;
            let headers = String::from_utf8_lossy(&writer.bytes[..split]);
            assert!(headers.contains("Content-Type: audio/pcm\r\n"));
            assert!(headers.contains("Transfer-Encoding: chunked\r\n"));
            assert!(headers.contains("X-Saragossa-Audio-Sample-Rate: 24000\r\n"));
            assert!(headers.contains("Trailer: X-Saragossa-TTS-Frame-Limit-Reached\r\n"));
            assert!(!headers.contains("Content-Length:"));
            let mut body = &writer.bytes[split..];
            let mut pcm = Vec::new();
            loop {
                let end = find_sub(body, b"\r\n").expect("invariant: taille chunk terminée");
                let size = usize::from_str_radix(
                    std::str::from_utf8(&body[..end]).expect("invariant: taille ASCII"),
                    16,
                )
                .expect("invariant: taille hexadécimale");
                body = &body[end + 2..];
                if size == 0 {
                    break;
                }
                assert_eq!(size % 2, 0);
                pcm.extend_from_slice(&body[..size]);
                assert_eq!(&body[size..size + 2], b"\r\n");
                body = &body[size + 2..];
            }
            assert_eq!(
                body,
                format!("X-Saragossa-TTS-Frame-Limit-Reached: {reached}\r\n\r\n").as_bytes()
            );
            let wav = mono_f32_to_wav(&samples, 24_000)?;
            let mut reader =
                WavReader::new(Cursor::new(wav)).expect("invariant: WAV encodé valide");
            let batch: Vec<u8> = reader
                .samples::<i16>()
                .flat_map(|s| s.expect("invariant: sample valide").to_le_bytes())
                .collect();
            assert_eq!(pcm, batch);
        }
    }
    Ok(())
}

#[test]
fn disconnect_and_timeout_stop_synthesis_without_second_response() -> ServeResult<()> {
    let mut reference = Vec::new();
    send_speech_stream(&mut reference, 24_000, |_| Ok(output(Vec::new(), false)))?;
    let header_len = find_sub(&reference, b"\r\n\r\n").expect("invariant: en-têtes écrits") + 4;
    for (fail_after, fail_flush, expected_callbacks) in [
        (Some(8), None, 0),
        (Some(header_len), None, 1),
        (Some(header_len + 4), None, 1),
        (None, Some(1), 0),
        (None, Some(2), 1),
        (None, Some(3), 2),
        (None, Some(4), 2),
    ] {
        let mut writer = Writer {
            fail_after,
            fail_flush,
            ..Writer::default()
        };
        let mut callbacks = 0;
        let mut finished = false;
        let result = send_speech_stream(&mut writer, 24_000, |on_audio| {
            for _ in 0..2 {
                callbacks += 1;
                on_audio(&[0.25, -0.25])?;
            }
            finished = true;
            Ok(output(Vec::new(), false))
        });
        // Même le gestionnaire externe ne doit pas ajouter de JSON au PCM.
        super::super::http::reply_on_error(&mut writer, result)?;
        assert_eq!(callbacks, expected_callbacks);
        assert_eq!(finished, fail_flush == Some(4));
        assert_eq!(
            writer.bytes.windows(5).filter(|s| *s == b"HTTP/").count(),
            1
        );
        if !finished {
            assert!(!writer
                .bytes
                .ends_with(b"0\r\nX-Saragossa-TTS-Frame-Limit-Reached: false\r\n\r\n"));
        }
    }
    Ok(())
}

#[test]
fn inference_failure_leaves_an_incomplete_chunked_body() -> ServeResult<()> {
    let mut bytes = Vec::new();
    let result = send_speech_stream(&mut bytes, 24_000, |on_audio| {
        on_audio(&[0.25])?;
        Err(ServeError::args("codec factice interrompu"))
    });
    super::super::http::reply_on_error(&mut bytes, result)?;
    assert!(bytes.ends_with(b"2\r\n\xff\x1f\r\n"));
    assert_eq!(bytes.windows(5).filter(|s| *s == b"HTTP/").count(), 1);
    Ok(())
}

#[test]
fn invalid_stream_option_is_rejected_before_synthesis() {
    let mut bytes = Vec::new();
    let result = handle_speech_with(
        &mut bytes,
        br#"{"input":"Bonjour","stream":"true"}"#,
        None,
        |_, _, _, _| unreachable!("invariant: requête invalide rejetée avant synthèse"),
    );
    assert!(result.is_err());
    assert!(bytes.is_empty());
}

#[cfg(unix)]
#[test]
fn closed_socket_does_not_start_synthesis() -> ServeResult<()> {
    let (mut server, client) =
        std::os::unix::net::UnixStream::pair().map_err(|e| ServeError::io("socket de test", e))?;
    drop(client);
    send_speech_stream(&mut server, 24_000, |_| {
        unreachable!("invariant: socket fermée détectée avant synthèse")
    })
}

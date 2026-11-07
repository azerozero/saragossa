"""Lecture commune du journal de résultats, récupérable après interruption."""

import json
import pathlib
import warnings


def read_results(path, *, repair_tail=False):
    """Ignore une seule ligne finale invalide sans fin de ligne, avec avertissement.

    Toute ligne invalide terminée ou suivie d'une autre ligne reste fatale.
    La reprise retire le fragment avant append ; l'analyse ne modifie rien.
    Une dernière ligne valide sans LF reçoit un LF avant reprise.
    """
    path = pathlib.Path(path)
    raw_lines = path.read_bytes().splitlines(keepends=True)
    rows, valid_bytes, truncated = [], 0, False
    for index, raw in enumerate(raw_lines):
        try:
            row = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(raw_lines) - 1 or raw.endswith((b"\n", b"\r")):
                raise ValueError(f"{path.name}: JSON invalide ligne {index + 1}") from None
            warnings.warn(f"{path.name}: dernière ligne JSON tronquée ignorée "
                          f"(ligne {index + 1})", RuntimeWarning, stacklevel=2)
            truncated = True
            break
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}: objet JSON attendu ligne {index + 1}")
        rows.append(row)
        valid_bytes += len(raw)
    if repair_tail and raw_lines:
        with path.open("r+b") as out:
            if truncated:
                out.truncate(valid_bytes)
            elif not raw_lines[-1].endswith(b"\n"):
                out.seek(0, 2)
                out.write(b"\n")
    return rows


def _self_check():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "results.jsonl"
        first = b'{"kind":"entry_done"}\n'
        for fragment in (b'{"kind":', b'{"kind":"part', b'{"kind":"\xc3'):
            original = first + fragment
            path.write_bytes(original)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                assert read_results(path) == [{"kind": "entry_done"}]
                assert len(caught) == 1 and "tronquée" in str(caught[0].message)
                assert path.read_bytes() == original
                assert read_results(path, repair_tail=True) == [{"kind": "entry_done"}]
            assert path.read_bytes() == first
            with path.open("ab") as out:
                out.write(b'{"kind":"run_start"}\n')
            assert len(read_results(path)) == 2
        for invalid in (first + b'{"kind":\n', b'{"kind":\n' + first,
                        first + b'{"kind":\n{"kind":', b'[]\n'):
            path.write_bytes(invalid)
            try:
                read_results(path, repair_tail=True)
            except ValueError:
                pass
            else:
                raise AssertionError("corruption acceptée")
            assert path.read_bytes() == invalid
        path.write_bytes(first.rstrip(b"\n"))
        assert read_results(path, repair_tail=True) == [{"kind": "entry_done"}]
        assert path.read_bytes() == first
        path.write_bytes(b"")
        assert read_results(path, repair_tail=True) == []


if __name__ == "__main__":
    _self_check()
    print("results: auto-test ok")

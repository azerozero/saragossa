"""CPU reference: sentence-transformers 6.3.0, transformers 5.22.2, torch 2.15.0.
pip install sentence-transformers==6.3.0 transformers==5.22.2 torch==2.15.0
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import local_model, provenance, write

TEXTS = ["Bonjour le monde.", "Le chat dort sur le canapé.", "Où est la gare ?",
         "Une recherche sémantique multilingue.", "La pluie tombe à Brest.",
         "Je ne suis pas d'accord.", "Été, Noël et cœur : accents français.",
         "42 kilomètres en 3 heures.", "", "🙂 Merci !",
         "Hello world.", "The quick brown fox jumps over the lazy dog.",
         "How do I reset my password?", "A database transaction must be atomic.",
         "This sentence is false.", "fn main() { println!(\"hello\"); }",
         "SELECT id FROM users WHERE active = true;", "def square(x): return x * x",
         "git diff --stat", "Rust Result<T, E> and Python exceptions."]


def main():
    import torch
    from sentence_transformers import SentenceTransformer, models
    torch.set_num_threads(4)
    path = local_model("embedder")
    # This cache lacks modules.json; explicitly reproduce E5 masked mean pooling.
    transformer = models.Transformer(str(path), max_seq_length=512,
        model_args=dict(local_files_only=True), tokenizer_args=dict(local_files_only=True))
    model = SentenceTransformer(modules=[transformer, models.Pooling(384, pooling_mode="mean"),
                                        models.Normalize()], device="cpu")
    cases = [dict(text=text, kind="query" if i % 2 == 0 else "passage") for i, text in enumerate(TEXTS)]
    vectors = model.encode([f'{c["kind"]}: {c["text"]}' for c in cases],
                           batch_size=4, normalize_embeddings=True, show_progress_bar=False)
    write("embedder", "e5", vectors.flatten().tolist(), "f32", [20, 384],
          provenance("embedder", path, ["sentence-transformers", "transformers", "torch"]),
          "cosine >= 0.9999; max_abs <= 0.001; normalized f32, masked mean pooling",
          cases=cases, pooling="mean", max_seq_length=512)


if __name__ == "__main__":
    main()

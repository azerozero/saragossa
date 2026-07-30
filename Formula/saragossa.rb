# Formule Homebrew pour saragossa.
#
# Ce fichier fait du dépôt un *tap* Homebrew. Installation :
#
#   brew tap azerozero/saragossa https://github.com/azerozero/saragossa
#   brew install saragossa             # build depuis le dernier tag (stable)
#   brew install --HEAD saragossa      # build depuis la branche main
#
# Deux variantes coexistent : `stable` (dernier tag `vX.Y.Z`, défaut) et `--HEAD`
# (branche `main`). Bumper `url` + `sha256` à chaque nouveau tag posé par
# release-plz.
class Saragossa < Formula
  desc "Pure-Rust Metal inference engine for Apple Silicon LLMs, STT and TTS"
  homepage "https://github.com/azerozero/saragossa"
  url "https://github.com/azerozero/saragossa/archive/refs/tags/v0.2.0.tar.gz"
  sha256 "dc5d8065c7b7da21e5a9242e07134d4620511e9a8a4bde6b15e2ade674fb9307"
  license "Apache-2.0"
  head "https://github.com/azerozero/saragossa.git", branch: "main"

  depends_on "rust" => :build
  depends_on arch: :arm64 # kernels Metal GPU → Apple Silicon uniquement
  depends_on :macos

  def install
    # Le binaire `saragossa` exige la feature `devtools`, active par défaut
    # (metal + devtools). `cargo install` la conserve donc sans réglage.
    system "cargo", "install", *std_cargo_args
  end

  def caveats
    <<~EOS
      saragossa compile ses kernels Metal au premier lancement
      (MTLDevice newLibraryWithSource). Cela requiert la Metal Toolchain :

        xcodebuild -downloadComponent MetalToolchain

      Les poids de modèle sont résolus depuis Hugging Face au premier `run` :

        saragossa run mlx-community/Qwen3-4B-4bit
        saragossa list
        saragossa serve --model-dir ~/models --api-key local-dev
    EOS
  end

  test do
    # Vérifie que le binaire se lance et expose ses sous-commandes (exit 0).
    assert_match "saragossa", shell_output("#{bin}/saragossa --help 2>&1")
  end
end

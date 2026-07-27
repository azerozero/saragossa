# Formule Homebrew pour saragossa.
#
# Ce fichier fait du dépôt un *tap* Homebrew. Installation :
#
#   brew tap azerozero/saragossa https://github.com/azerozero/saragossa
#   brew install --HEAD saragossa      # build depuis la branche main
#
# La variante `stable` (build depuis une release taguée, sans `--HEAD`) est
# commentée plus bas : elle s'active dès qu'un tag `vX.Y.Z` + tarball existe.
# Aujourd'hui la CI release-plz ne pose aucun tag (secret RELEASE_PLZ_TOKEN
# absent) → seule `--HEAD` fonctionne. Décommenter `url`/`sha256` au premier tag.
class Saragossa < Formula
  desc "Pure-Rust Metal inference engine for Apple Silicon LLMs, STT and TTS"
  homepage "https://github.com/azerozero/saragossa"
  license any_of: ["MIT", "Apache-2.0"]
  head "https://github.com/azerozero/saragossa.git", branch: "main"

  # stable do
  #   url "https://github.com/azerozero/saragossa/archive/refs/tags/v0.1.0.tar.gz"
  #   sha256 "<à remplir au 1er tag>"
  # end

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

"""CPU reference: transformers 5.22.2. Llama 3.1/3.3 : date figée du template.

Contrairement à 3.2 (`strftime_now`), le template 3.1/3.3 fixe `date_string` à
« 26 Jul 2024 » par défaut. Miroir mlx-community : le dépôt Meta est gated.
Run with python -I.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TEMPLATE_PACKAGES, require_versions, local_model, provenance, write_json


def main():
    require_versions(TEMPLATE_PACKAGES)
    from transformers import AutoTokenizer
    path = local_model("llama31")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    messages = [dict(role="system", content="Réponds brièvement."),
                dict(role="user", content="Bonjour !"),
                dict(role="assistant", content="Bonjour, comment puis-je aider ?"),
                dict(role="user", content="What is 2 + 2?")]
    cases = []
    for generation in [False, True]:
        case = dict(name=f"multi/generation={generation}", messages=messages, tools=[],
                    add_generation_prompt=generation, enable_thinking=None)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=generation)
        case.update(rendered=rendered, token_ids=tokenizer.encode(rendered, add_special_tokens=False))
        cases.append(case)
    write_json("chat_templates", "llama31", dict(model="llama31", cases=cases),
               provenance("llama31", path, TEMPLATE_PACKAGES), "exact UTF-8")


if __name__ == "__main__":
    main()

"""CPU reference: transformers 5.22.2. Mistral v0.3 : message système sans dernier tour user.

Le template officiel n'injecte le système que si `loop.last` : avec
[system, user, assistant], il est donc absent du rendu. Run with python -I.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TEMPLATE_PACKAGES, require_versions, local_model, provenance, write_json


def main():
    require_versions(TEMPLATE_PACKAGES)
    from transformers import AutoTokenizer
    path = local_model("mistral7b")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    messages = [dict(role="system", content="Réponds brièvement."),
                dict(role="user", content="Bonjour !"),
                dict(role="assistant", content="Bonjour, comment puis-je aider ?")]
    cases = []
    for generation in [False, True]:
        case = dict(name=f"system_user_assistant/generation={generation}", messages=messages,
                    tools=[], add_generation_prompt=generation, enable_thinking=None)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=generation)
        case.update(rendered=rendered, token_ids=tokenizer.encode(rendered, add_special_tokens=False))
        cases.append(case)
    write_json("chat_templates", "mistral7b_tail", dict(model="mistral7b", cases=cases),
               provenance("mistral7b", path, TEMPLATE_PACKAGES), "exact UTF-8")


if __name__ == "__main__":
    main()

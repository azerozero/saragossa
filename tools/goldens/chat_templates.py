"""CPU reference: transformers 5.22.2. pip install transformers==5.22.2

Local tokenizers only; no weights, network or accelerator. Run with python -I.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import local_model, provenance, write_json


def main():
    from transformers import AutoTokenizer
    multi = [dict(role="system", content="Réponds brièvement."),
             dict(role="user", content="Bonjour !"),
             dict(role="assistant", content="Bonjour, comment puis-je aider ?"),
             dict(role="user", content="What is 2 + 2?")]
    tools = [dict(type="function", function=dict(name="weather", description="Weather",
             parameters=dict(type="object", properties=dict(city=dict(type="string")), required=["city"])))]
    calls = [dict(role="system", content="Utilise les outils."),
             dict(role="user", content="Météo à Paris ?"),
             dict(role="assistant", content="", tool_calls=[dict(id="call_1", type="function",
                  function=dict(name="weather", arguments=dict(city="Paris")))]),
             dict(role="tool", name="weather", tool_call_id="call_1", content='{"temperature":18}'),
             dict(role="user", content="Merci, résume.")]
    for name in ["qwen38", "qwen36_moe", "gemma3", "gemma4", "qwen3_tts"]:
        path = local_model(name)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        template = tokenizer.chat_template
        if not template:
            # TTS has a speech-specific prompt, not an apply_chat_template contract.
            write_json("chat_templates", name, dict(model=name, cases=[], unavailable="no chat template"),
                       provenance(name, path, ["transformers"]), "exact UTF-8")
            continue
        cases = []
        thinking_modes = [False, True] if "enable_thinking" in str(template) else [None]
        for kind, messages, definitions in [("multi", multi, []), ("tools", calls, tools)]:
            for generation in [False, True]:
                for thinking in thinking_modes:
                    kwargs = {} if thinking is None else dict(enable_thinking=thinking)
                    case = dict(name=f"{kind}/generation={generation}/thinking={thinking}",
                                messages=messages, tools=definitions, add_generation_prompt=generation,
                                enable_thinking=thinking)
                    try:
                        rendered = tokenizer.apply_chat_template(messages, tools=definitions or None,
                            tokenize=False, add_generation_prompt=generation, **kwargs)
                        case.update(rendered=rendered,
                                    token_ids=tokenizer.encode(rendered, add_special_tokens=False))
                    except Exception as error:
                        # Unsupported tool roles are reference results, never invented text.
                        case["error"] = f"{type(error).__name__}: {error}"
                    cases.append(case)
        write_json("chat_templates", name, dict(model=name, cases=cases),
                   provenance(name, path, ["transformers"]), "exact UTF-8; Gemma3 BOS supplied by tokenizer")


if __name__ == "__main__":
    main()

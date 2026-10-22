"""IFEval checker (google-research instruction_following_eval, as vendored by lm-evaluation-harness, Apache-2.0).
evaluate_instruction_following(prompt, response, instruction_id_list, kwargs, strict=True) -> bool (prompt-level:
every instruction followed).  Deps: nltk (punkt_tab), langdetect, immutabledict."""
from . import instructions_registry


def _check(response, instruction_id_list, kwargs):
    for iid, kw in zip(instruction_id_list, kwargs):
        ins = instructions_registry.INSTRUCTION_DICT[iid](iid)
        ins.build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
        if not (response.strip() and ins.check_following(response)):
            return False
    return True


def evaluate_instruction_following(prompt, response, instruction_id_list, kwargs, strict=True):
    if strict:
        return _check(response, instruction_id_list, kwargs)
    r = response.strip()
    lines = r.split("\n")
    variants = [r, "\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(), "\n".join(lines[1:-1]).strip()]
    variants += [v.replace("*", "") for v in variants]
    return any(_check(v, instruction_id_list, kwargs) for v in variants)

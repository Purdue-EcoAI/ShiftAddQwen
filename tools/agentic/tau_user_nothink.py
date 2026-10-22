"""Patch tau-bench (run once after cloning) so the LLM user simulator runs without thinking: adds
extra_body={"chat_template_kwargs": {"enable_thinking": False}} to its litellm completion call (llama-server honours it).
The agent under test keeps thinking on.   python tools/agentic/tau_user_nothink.py <path/to/tau-bench>"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "tau_bench/envs/user.py"; s = p.read_text()
old = "model=self.model, custom_llm_provider=self.provider, messages=messages\n"
if "enable_thinking" in s: print("already patched"); sys.exit()
assert old in s
p.write_text(s.replace(old, old.rstrip("\n") + ',\n            extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # user simulator without thinking\n'))
print("patched", p)

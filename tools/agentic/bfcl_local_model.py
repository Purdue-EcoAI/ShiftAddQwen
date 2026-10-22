"""Register a llama-server-backed FC model in BFCL (run once after cloning gorilla):
    python tools/agentic/bfcl_local_model.py <path/to/berkeley-function-call-leaderboard>
Adds "llama-server-FC": OpenAI chat-completions handler with tools (dots in function names are sent as underscores
and mapped back by the checker: underscore_to_dot=True), model name "default" (llama-server --alias
default); OPENAI_BASE_URL points at the server, so the GGUF's own chat template and tool-call parser are used.
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "bfcl_eval/constants/model_config.py"
s = p.read_text()
if "llama-server-FC" in s:
    print("already registered"); sys.exit()
entry = '''    "llama-server-FC": ModelConfig(
        model_name="default",
        display_name="llama-server (FC)",
        url="https://github.com/ggml-org/llama.cpp",
        org="local",
        license="apache-2.0",
        model_handler=OpenAICompletionsHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,
    ),
'''
anchor = '    "openbmb/MiniCPM-SALA-FC": ModelConfig('
assert anchor in s
p.write_text(s.replace(anchor, entry + anchor, 1))
print("registered llama-server-FC in", p)

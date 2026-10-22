"""Patch tau-bench (run once after cloning) so the checkpoint / result file name includes the process id: two runs
started in the same second otherwise share a file and overwrite each other's episodes.
  python tools/agentic/tau_ckpt_pid.py <path/to/tau-bench>"""
import sys, pathlib, re
p = pathlib.Path(sys.argv[1]) / "tau_bench/run.py"; s = p.read_text()
if "os.getpid()" in s: print("already patched"); sys.exit()
m = re.search(r'ckpt_path = f"(.*?)\.json"', s); assert m, "checkpoint line not found"
s = s.replace(m.group(0), m.group(0)[:-6] + '_{os.getpid()}.json"'); p.write_text(s); print("patched", p)

"""Write the two KLD evaluation texts: work/data/wiki.test.raw (wikitext-2 test) and mix.test.raw."""
import subprocess, sys
from pathlib import Path
from datasets import load_dataset
from saq.model import WORK
(WORK / "data").mkdir(parents=True, exist_ok=True)
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
(WORK / "data/wiki.test.raw").write_text("".join(ds["text"]))
subprocess.run([sys.executable, str(Path(__file__).parent / "make_mix_text.py")], check=True)

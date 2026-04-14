from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent.parent / "04_probe_word2vec.py"),
    run_name="__main__",
)

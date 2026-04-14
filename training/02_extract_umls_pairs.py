from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent.parent / "02_extract_umls_pairs.py"),
    run_name="__main__",
)

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent.parent / "03_align_ntxent.py"),
    run_name="__main__",
)

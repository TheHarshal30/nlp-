from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent.parent / "05_probe_diagnostics.py"),
    run_name="__main__",
)

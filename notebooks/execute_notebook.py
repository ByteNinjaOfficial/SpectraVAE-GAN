"""
Execute 01_EDA.ipynb and save all outputs directly inside the notebook.
"""

from pathlib import Path
import nbformat
from nbclient import NotebookClient

NOTEBOOK_PATH = Path(__file__).resolve().parent / "01_EDA.ipynb"

def run():
    print(f"[INFO] Loading notebook: {NOTEBOOK_PATH}")
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)

    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}}
    )

    print("[INFO] Executing all cells in 01_EDA.ipynb...")
    client.execute()

    print(f"[INFO] Execution complete. Writing executed notebook to {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print("[SUCCESS] 01_EDA.ipynb executed and saved with all outputs embedded!")

if __name__ == "__main__":
    run()

from pathlib import Path
import importlib.util
import sys


ROOT_APP = Path(__file__).resolve().parents[1] / "app.py"
ROOT_DIR = str(ROOT_APP.parent)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

spec = importlib.util.spec_from_file_location("root_streamlit_app", ROOT_APP)
root_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(root_app)


if __name__ == "__main__":
    root_app.main()

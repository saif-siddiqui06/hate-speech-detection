from pathlib import Path
import importlib.util


ROOT_APP = Path(__file__).resolve().parents[1] / "app.py"
spec = importlib.util.spec_from_file_location("root_streamlit_app", ROOT_APP)
root_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(root_app)


if __name__ == "__main__":
    root_app.main()

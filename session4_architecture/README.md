# Architecture Lab (Session 4)

Keep `app.py` and `arch_engine.py` together in ONE folder, separate from the Session 3 app (both are named app.py).

    cd architecture_lab
    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install streamlit pandas
    streamlit run app.py

You only run `app.py`. `arch_engine.py` is imported by it and must sit beside it; do not run it on its own.

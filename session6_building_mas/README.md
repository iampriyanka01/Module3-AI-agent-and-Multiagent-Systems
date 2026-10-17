# Building MAS Lab (Session 6, cont.): Building Multi-Agent Systems with LangChain and CrewAI

Keep `app.py` and `crew_engine.py` together in ONE folder.

    pip install streamlit pandas
    streamlit run app.py

You only run `app.py`. `crew_engine.py` is imported by it and must sit beside it; do not run it on its own.

- **Demo mode** (default) needs no API key, and needs neither LangChain nor CrewAI installed. The agents are scripted and all data (packages, prices, search results) is pretend.
- **Live mode** (Gemini) is optional: it lets Gemini write the text on the CrewAI-style labs (pages 8 and 9). It was not tested end to end.
- The "mini CrewAI" on pages 7 to 9 is a small model written for this lab to show what a framework does for you. It is not the CrewAI library; the real library has more options and may differ in details.
- The notebook's code is shown in expanders. Check the LangChain version notes on page 1 against your own install.

Pages: 1 one agent (tool + memory) · 2 centralized desk · 3 decentralized desk · 4 hybrid desk · 5 three designs compared · 6 LangChain vs CrewAI · 7 CrewAI building blocks · 8 sequential process · 9 hierarchical process · 10 review.

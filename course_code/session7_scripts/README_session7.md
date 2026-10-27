# Session 7: The Three C's in Multi-Agent Systems

- `Agents_Session_7_Three_Cs_and_Design_Patterns_in_CrewAI.ipynb`: the session notebook. Works in **Google Colab** and in **VS Code**.
- `scripts/`: the same five demos as standalone Python scripts, for running from a terminal.

| Demo | Notebook section | Script |
|---|---|---|
| All three communication patterns in one crew | Demo 1 | `scripts/crewai_comm_pattern.py` |
| Manager / worker | Demo 2 | `scripts/manager_worker.py` |
| Sequential pipeline | Demo 3 | `scripts/sequential_pipeline.py` |
| Peer / swarm | Demo 4 | `scripts/swarm_peer.py` |
| Federated teams / specialist agents | Demo 5 | `scripts/federated_teams_specialist_agents.py` |

You only need a **Gemini API key**. The Wikipedia tool is free and needs no key.

## In Google Colab

1. Upload the notebook (File > Upload notebook).
2. Save your key once: click the **key icon** in the left sidebar, add a secret named `GOOGLE_API_KEY`, and switch on **Notebook access**.
3. Run the cells from the top. If Colab shows "Restart session" after the install, click it and continue from the next cell.

## In VS Code

1. Install Python **3.10 to 3.13** (CrewAI doesn't support 3.14 yet), VS Code, and the **Python** and **Jupyter** extensions.
2. In a terminal in this folder:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac or Linux
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and put your key in it.
4. Open the notebook, choose the `.venv` kernel (top right), and run the cells. Or run a script: `cd scripts` then, for example, `python manager_worker.py` (put a copy of `.env` in `scripts/`, or type the key when asked).

## Good to know

- Each demo makes several Gemini calls, limited to 4 a minute (`max_rpm=4`) to stay within the free tier, so some take a few minutes.
- In the notebook, crews run with `await crew.kickoff_async()`. In the scripts, it's `crew.kickoff()`. Both are correct for where they run.
- `verbose=True` prints long logs on purpose, so you can watch managers delegate and peers ask each other. The answer is at the bottom.

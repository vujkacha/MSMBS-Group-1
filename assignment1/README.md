# Epidemiological Model Assignment — Parameter Exploration

**Course**: KEN3170 — Multi-scale modeling of biological systems
**Group number**: [1]

---

## 1. Repository overview
- `analysis.ipynb` — main notebook containing all required sections (Setup, Part 1–3, Conclusions)
- `requirements.txt` — Python dependencies (numpy, matplotlib, pandas, scipy, seaborn)
- `README.md` — this file

**How to run**: 
1. into the terminal, write `pip install -r requirements.txt` 
2. open and run `analysis.ipynb` top to bottom

---

## 2. Part 1 — Parameter analysis function
**Function**: `analyze_recovery_rates(beta, mu, N, I0, simulation_days)`
- Approach: SIRD model modified accordingly from the computer practical, so that it runs on the parameters of the given function, that being beta, mu, N, I0 and simulation_days for every gamma value. Finally, all required datapoints (R0, peak infected, peak day and total deaths) are added for each gamma case. Lastly, the graph is created with a function for each gamma case.
- Output DataFrame (γ = 0.05–0.25), matching the notebook exactly.

---

## 3. Part 2 — Scenario comparison
- Result tables for Scenario A (High Transmission) and Scenario B (Low Transmission)
- Which scenario is worse for public health, and why

---

## 4. Part 3 — Policy recommendations
- 4.1 Parameter impact analysis
- 4.2 Intervention analysis
- 4.3 Real-world application

---

## 5. Conclusions
We have explored a different way to simulate an SIRD model and tested it with various parameters, helping us evaluate the impact of each disease scenario. Lastly, we have analysed the impact of parameters, intervention, and have elaborated on the consequences of a real-life scenario in relation to this.
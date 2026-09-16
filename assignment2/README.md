Assignment 2 - Metabolic modeling

Group number - 1

1. Repository overview 
- `analysis.ipynb` — main notebook containing all required sections (Setup, Part 1–3, Conclusions)
- `requirements.txt` — Python dependencies (numpy, matplotlib, pandas, scipy, seaborn)
- `README.md` — this file
- e_coli_core_expression.csv - csv file gain for this assignment 
- e_coli_core.json - file with main e_coli data from tutorial 
- saved_map.png photo of the reactions

**How to run**: 
1. into the terminal, write `pip install -r requirements.txt` 
2. open and run `analysis.ipynb` top to bottom


2. Ex 1 displayed the map saved_map.png
3. Ex 2 enzyme acitivty:
    - reversible reactions: lower bound = - value upper bound = + value
    - irreversible reactions: lower bound = 0 upper bound = value 
    - No data = we left at the default
    - ATPM lower bound left unchanged 
    - EX_gl__D_e pre existing tight bound removed
4. Ex 3 FBA optimization:
    - Maximal biomass production rate under the enzyme-activity constraints
    - Absolute flux bound of 5 mmol/gDW/h re-established on EX_glc__D_e
    - Maximal biomass production rate under the additional glucose constraint
5. Ex 4 Glucose uptake:
    - Plot of glucose exchange and biomass production rate 
    - Graph analyze, limit of growth 
    - Determination of what reaction becomes active in 2nd segment 

Conclusion
We compared an unconstrained E. coli core model against an enzyme-activity-constrained version built from gene expression data, and found that applying real enzymatic capacity limits sharply lowers the predicted maximal growth rate to biologically realistic values. Sweeping the glucose uptake bound revealed three distinct growth regimes: linear aerobic growth, a transition where acetate secretion (EX_ac_e) begins once the respiratory chain  saturates, and a final plateau where actual glucose usage is capped by the transporter's own enzyme limit 

Assignment 3 - Biological Networks

Group Number 1

1. Repository overview:
    1 KEN3170_BooleanModelling... - is a main notebook to run with all code 
    2 README.md - read me file with answer for 3 questions 
    3 requirements.txt is a txt requirement file with just requirements


2. How to run this notebook
    1. into the terminal, write `pip install -r requirements.txt` 
    2. open and run KEN3170 notebook top to bottom

3. Answer for all questions in teh Assignment
    1. Question: Which mutation is most dangerous and why? Provide quantitative evidence.
       Answer: The A, B and C are equally dangerous, gamaged ceells keep growing 50% in all three against 3.1% in normal. But A reached canser state faster, within 4 steps. D is the safest - damaged cells die, 0% cancer like
    2. Question: Explain the role of feedback loops (e.g., MYC → MDM2 → p53)
       Answer: Main loop p53 - > Blocking MYC - > MYC turns on MDM2 - > blocks p53
       if p53 is ON - cell dies, if OFF cell grows. 
    3. Question: What are the limitations of this Boolean network model? Discuss 3 specific limitations.
       Answer: 
       1. First limitation is On/OFF state, genes are more complex and mutations can be partial. 
       2. Timing - all gene updates occur at the same time point while in real life it is different 
       3. Cancer is also more complex and it needs couple of mutations together, but out model uses only 1 at the time



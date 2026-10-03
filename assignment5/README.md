# Assignment 5 - Plant systems biology

1. In the initial phase, the cells are visible with the pathogen attached to the left. As time progresses, the color coded infection forms around the pathogen (light purple) and progressively spreads deeper into multiple cells. A significant wall deformation is visible in the color coded cells (cyan for normal and light purple for infected). This may be the case as the pathogen can spread wall weakening chemicals, resulting in these visible deformations.

2. The function checks for two conditions on an arbitrary cell before proceeding to weaken its cell walls. It checks whether the pathogen chemical level is sufficient (at least more than 0.1 and could be at most 1.2) and whether the cell in question is not the pathogen itself (cell type 2). If the conditions are fulfilled, the wall stiffness (defaulted at 3) is reduced as follows: wall stiffness = 3 - pathogen_chemical_level. On the other hand, if the pathogen chemical level is insufficient or the cell is the pathogen itself, the wall stiffness is reassigned to its default value 3. Moreover, the beginning of the function allows the pathogen to enlarge its target area. Once the area of the pathogen bypasses the required threshold, it divides. This gives it a distinct behavior from the other cells.

3. The diffusion coefficient is inversely proportional to wall stiffness (0.00001 / stiffness), provided that the stiffness is above 0.001 to avoid division by 0. The mechanism where chemical lowers stiffness, which raises diffusion and faster diffusion spreads the chemical, is a positive feedback loop, as in the next operation an increased flux of chemical will raise diffusion further into adjacent cells, amplifying the system rather than compensating for the change. The limit would be when the flux levels off at low stiffness, and concentration gradients across neighboring cells balance out.



6. Since the defense mechanism alters the wall stiffness based on chemical concentration, the new logic would go to the section “//cell wall weakening happens here” under IF block of stiffness_inf:
Calculate chemical concentration (patho_chem_level)
Define normal baseline stiffness
Define a threshold

if patho_chem_level > threshold:
    Calculate high stiffness (baseline stiffness + extra stiffness)
    Apply high stiffness to all cell walls
    Set cell veto to true
else if patho_chem_level > 0.1 AND cell is not type 2:
    Calculate low stiffness (baseline stiffness - patho_chem_level)
    Apply low stiffness to all cell walls
    Set cell veto to false
else:
    Apply baseline stiffness to all cell walls
    Set cell veto to true
Instead of only checking if the chemical is greater than 0.1, we check if the chemical has reached a threshold. It is a negative feedback loop as high stiffness decreases siffusion coefficient and therefore the system opposes back to stabilize itself, compensating for the initial increase.

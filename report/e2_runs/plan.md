I am now writing the last chapter of my report - focused on the e2 rl experiments. This is going to be about both the multi-fidelity plus speed up from threading to the actual runs run so far - the results of those and what are my next experiments/steps with an evaluation. THis will be about 15 pages - with a strong focus on evaluation. write now we're drafting a plan - the e2 runs and stuff have got quite messy so we need to organise everything and explain all my decisions and thought processes. Especially as there will be a lot of red herring and fluff from the failed e2 runs on the buggy simulator (which has now been fixed). Exploring the repository this is what i see as important rn:


Overall there have been two successful new runs.

1. Multi-fidelity section:
 - background lit - need to get the citations in order - there has been some background section written somewhere just not sure where.
 - implementation details (`report/e2_runs/multi_fidelity.md` - i think i have implemented stage A - the lookahead) but there are also multiple stages of this - need to be clear about what current implementation does.
 - I have explained water coarsening but not water caching - that can be explained here (implementation note for when we go down to 5 - we will have to keep the same positions to not have to redo water caching)
    - caching figures/investigation to include:
        - have a look at project_context/meeting_notes/M6.md and the files it references.
 - result of run: `report/e2_runs/mf_runA_results.md` - need to be very clear as to what the design decisions were.
  - e.g. for log ti action space - have to say that came from previous failed runs as better for bottom spheres which had high error - didn't have time to re-check that.
  - the config.json for these should be great for reproducability / knowing what exactly i ran it with
  - different figures properly explained and what my diagnostic / eval scripts actually do
  - section on noise selection plus evaluation of amount of noise during the different series

For now lets just focus on getting this section polished and ready and seeing which runs we should run after this.

Have a todo section and a future work section - todo we will get to the other is for more broad changes. make todo section - include investigation as to why the switching mechanism for 

Also what is the ROI used? as during investigation into gibbs ringing did kind of decide to use ROI=1 instead of 0 so an abalation run could be a good idea on that front...

explore the code base and results of runs thoroughly then create a new .md in this folder with all of that - feel free to come back to me for questions - trust the code more than .md s as some are out of date.




2. Baseline comparison:
 - cr_explainer.md (`src/baselines/cr_optimal.jl`)
 - including the alpha version for DOF of alpha (`ALPHA_DOF.md`)
 - TODO FOR ME: i need to understand this code
3. fitter code
 - Need to understand where we get the confidence estimates from
 - plus rest of this code (`report/mri_system_phantom/fits.md` a lot is in here and has been turned into )

3. TODO:
 - run multi-fidelity with the 5 spheres to see genuine adaptivity
 - run a confidence vs. no confidence ablation on the 5 spheres - see if that leads to better accuracy. Natural follow up is which confidence??
 - implement phase 2 of the mf plan - multi sampling for maximum concurrency.



4. Future work
 - Pareto curve - add a weighting - what kind of runs would that involve? 
    - For the 5 sphere adaptive one will probably have better results than all 14
 - T1/T2 joint parameter testing
 - expanded action space - multiple DOF 


First step - look at all the runs we have - which are missing - as they will take more time than writing so we should launch them as have only got 3 days.
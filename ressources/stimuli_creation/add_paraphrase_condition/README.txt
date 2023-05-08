I saved the Pereira stimset to a csv using a Jupyter notebook in my perturbed dir 
/om/user/gretatu/perturbed-neural-nlp/standalone_scripts/stimuli_creation/create-semantic-manipulations.ipynb


This saved: Pereira2018_stimulus_set_20220209.csv

Which I then worked on locally (using my normal control env, just using pandas)



The final stimuli df was saved in /chatGPT_stimuli_dfs/stimuli_chatGPT.pkl on 20230306 using the last section of create-semantic-manipulations.ipynb. 


Stats:

Min number of absolute word difference between original and paraphrase: 0
Max number of absolute word difference between original and paraphrase: 3
Mean number of absolute word difference between original and paraphrase: 1.00
Median number of absolute word difference between original and paraphrase: 1.0
Std number of absolute word difference between original and paraphrase: 0.95
Min number of word difference between original and paraphrase: -3
Max number of word difference between original and paraphrase: 3
Mean number of word difference between original and paraphrase: -0.44
Median number of word difference between original and paraphrase: 0.0
Std number of word difference between original and paraphrase: 1.31
Number of sentences that were manually replaced: 111
Mean/median fraction of overlapping words between original and paraphrase: 0.31/0.32
SD fraction of overlapping words between original and paraphrase: 0.08 and min/max fraction of overlapping words between original and paraphrase: 0.05/0.50


The overlap fraction was quantified as:
intersection(Num unique words in original sent) + (num unique words in paraphrased sent)
Divided by:
(Num unique words in original sent) + (num unique words in paraphrased sent)

(This used sentences that were stripped for all punctuation including final periods etc. to make the overlap count more fair and also count last words etc).



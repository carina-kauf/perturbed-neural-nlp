"""Script for compiling the paraphrased stimuli from chatGPT into a single csv file after manual look-over.
"""

import pandas as pd
import numpy as np
import re
import os

stimuli_df_copy = pd.read_csv('chatGPT_csvs/Pereira2018_stimulus_set_20220209.csv')
stimuli_gpt = pd.read_csv('chatGPT_csvs/chatGPT_perturbedNLP_paraphrase_compiled_for-manual-selection_GT_CK_merge.csv', encoding = "ISO-8859-1", index_col=0)

### SANITY CHECKS ###
# First, check whether the sentence_original col in stimuli_gpt matches the sentence col in stimuli_df_copy
assert(stimuli_gpt.sentence_original.tolist() == stimuli_df_copy.sentence.tolist())
# also check stimulus_id
assert(stimuli_gpt.stimulus_id.tolist() == stimuli_df_copy.stimulus_id.tolist())

### CLEAN UP THE GPT DF ###
# Remove sentence_paraphrase_{1,2,3} cols and sentence_paraphrase_{1,2,3}_stripped and manual_candidate_choice
stimuli_gpt = stimuli_gpt.drop(columns=['sentence_paraphrase_1', 'sentence_paraphrase_2', 'sentence_paraphrase_3',
                                        'sentence_paraphrase_1_stripped', 'sentence_paraphrase_2_stripped', 'sentence_paraphrase_3_stripped',
                                        'manual_candidate_choice', 'diff_sentence_length_original_paraphrase',
                                        'sentence_paraphrase_1_length', 'sentence_paraphrase_2_length', 'sentence_paraphrase_3_length',
                                        'sentence_paraphrase_closest_candidate_num'])
# Make manual_replacement nan into False and 1 into True
stimuli_gpt['manual_replacement'] = stimuli_gpt['manual_replacement'].replace({np.nan: False, 1: True})

# For all sentences that have manual_replacement == False, check that sentence_paraphrase_post_inspection is equal to sentence_paraphrase_closest
for i, row in stimuli_gpt.iterrows():
    if row['manual_replacement'] == False:
        if not (row['sentence_paraphrase_post_inspection'] == row['sentence_paraphrase_closest']):
            print(f'Row {i} has manual_replacement == False, but sentence_paraphrase_post_inspection != sentence_paraphrase_closest. Please check manually.')

# For all sentences that have manual_replacement == True, check that sentence_paraphrase_post_inspection is not equal to sentence_paraphrase_closest
for i, row in stimuli_gpt.iterrows():
    if row['manual_replacement'] == True:
        if row['sentence_paraphrase_post_inspection'] == row['sentence_paraphrase_closest']:
            print(f'Row {i} has manual_replacement == True, but sentence_paraphrase_post_inspection == sentence_paraphrase_closest. Please check manually.')


#### STATS ####
# Count number of words of sentence_original and sentence_paraphrase_post_inspection (len(sent.split(' ')))
stimuli_gpt['sentence_original_num_words'] = stimuli_gpt['sentence_original'].apply(lambda sent: len(sent.split(' ')))
stimuli_gpt['sentence_paraphrase_post_inspection_num_words'] = stimuli_gpt['sentence_paraphrase_post_inspection'].apply(lambda sent: len(sent.split(' ')))

# Get abs diff
stimuli_gpt['diff_abs_sentence_length_original_paraphrase'] = abs(stimuli_gpt['sentence_original_num_words'] - stimuli_gpt['sentence_paraphrase_post_inspection_num_words'])

# Get diff, not abs
stimuli_gpt['diff_sentence_length_original_paraphrase'] = stimuli_gpt['sentence_original_num_words'] - stimuli_gpt['sentence_paraphrase_post_inspection_num_words']


# Strip, lower-case the sentence_paraphrase_post_inspection
stimuli_gpt['sentence_paraphrase_post_inspection_stripped'] = stimuli_gpt['sentence_paraphrase_post_inspection'].apply(lambda sent: re.sub(r'[^\w\s]','',sent).lower())
assert(stimuli_gpt['diff_abs_sentence_length_original_paraphrase']).max() == 3
# assert that no sentences are identical after stripping and lower-casing
assert(stimuli_gpt['sentence_paraphrase_post_inspection_stripped'].values not in stimuli_gpt['sentence_original_stripped'].values)


# Abs difference: Print min, max, mean, median, std of diff_abs_sentence_length_original_paraphrase
print(f'Min number of absolute word difference between original and paraphrase: {stimuli_gpt["diff_abs_sentence_length_original_paraphrase"].min()}')
print(f'Max number of absolute word difference between original and paraphrase: {stimuli_gpt["diff_abs_sentence_length_original_paraphrase"].max()}')
print(f'Mean number of absolute word difference between original and paraphrase: {stimuli_gpt["diff_abs_sentence_length_original_paraphrase"].mean():.2f}')
print(f'Median number of absolute word difference between original and paraphrase: {stimuli_gpt["diff_abs_sentence_length_original_paraphrase"].median()}')
print(f'Std number of absolute word difference between original and paraphrase: {stimuli_gpt["diff_abs_sentence_length_original_paraphrase"].std():.2f}')

# Difference: Print min, max, mean, median, std of diff_sentence_length_original_paraphrase
print(f'Min number of word difference between original and paraphrase: {stimuli_gpt["diff_sentence_length_original_paraphrase"].min()}')
print(f'Max number of word difference between original and paraphrase: {stimuli_gpt["diff_sentence_length_original_paraphrase"].max()}')
print(f'Mean number of word difference between original and paraphrase: {stimuli_gpt["diff_sentence_length_original_paraphrase"].mean():.2f}')
print(f'Median number of word difference between original and paraphrase: {stimuli_gpt["diff_sentence_length_original_paraphrase"].median()}')
print(f'Std number of word difference between original and paraphrase: {stimuli_gpt["diff_sentence_length_original_paraphrase"].std():.2f}')

# Get stats on how many were manually replaced
print(f'Number of sentences that were manually replaced: {stimuli_gpt["manual_replacement"].sum()}')

# Get stats on how many words are identical between original and paraphrase
# Quantify unique overlapping words divided by unique words in both sentences (i.e. 1 means all words are identical)
# First, get unique words in original and paraphrase
stimuli_gpt['sentence_original_unique_words'] = stimuli_gpt['sentence_original_stripped'].apply(lambda sent: set(sent.split(' ')))
stimuli_gpt['sentence_paraphrase_post_inspection_unique_words'] = stimuli_gpt['sentence_paraphrase_post_inspection_stripped'].apply(lambda sent: set(sent.split(' ')))
# Then, get unique overlapping words
stimuli_gpt['sentence_original_unique_words_overlap'] = stimuli_gpt.apply(lambda row: row['sentence_original_unique_words'].intersection(row['sentence_paraphrase_post_inspection_unique_words']), axis=1)
# Then, get unique overlapping words divided by unique words in both sentences
stimuli_gpt['sentence_original_unique_words_overlap_ratio'] = stimuli_gpt.apply(lambda row: len(row['sentence_original_unique_words_overlap']) / (len(row['sentence_original_unique_words']) + len(row['sentence_paraphrase_post_inspection_unique_words'])), axis=1)
stimuli_gpt['sentence_original_unique_words_unique_overlap_ratio'] = stimuli_gpt.apply(lambda row: len(row['sentence_original_unique_words_overlap']) / (np.unique(list(row['sentence_original_unique_words']) + list(row['sentence_paraphrase_post_inspection_unique_words']))).shape[0], axis=1)


# Print min, max, mean, median, std of sentence_original_unique_words_overlap_ratio
print(f'Mean/median fraction of overlapping words between original and paraphrase: {stimuli_gpt["sentence_original_unique_words_overlap_ratio"].mean():.2f}/{stimuli_gpt["sentence_original_unique_words_overlap_ratio"].median():.2f}')
print(f'SD fraction of overlapping words between original and paraphrase: {stimuli_gpt["sentence_original_unique_words_overlap_ratio"].std():.2f} and min/max fraction of overlapping words between original and paraphrase: {stimuli_gpt["sentence_original_unique_words_overlap_ratio"].min():.2f}/{stimuli_gpt["sentence_original_unique_words_overlap_ratio"].max():.2f}')

## Save
fname_save = 'Pereira2018_stimulus_set_chatGPT.csv'
if not os.path.exists(f'chatGPT_csvs/{fname_save}'):
    stimuli_gpt.to_csv(f'chatGPT_csvs/Pereira2018_stimulus_set_chatGPT.csv', index=False)
    print(f'Saved {fname_save} to chatGPT_csvs/')
else:
    print(f'{fname_save} already exists in chatGPT_csvs/')

# Note that the final column of interest is 'sentence_paraphrase_post_inspection_stripped'

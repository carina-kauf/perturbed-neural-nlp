"""
This script loads the paraphrased chatGPT sentences and filters for length

"""
import pandas as pd
import numpy as np
import re


stimuli_df_copy = pd.read_csv('chatGPT_csvs/Pereira2018_stimulus_set_20220209.csv')
stimuli_gpt = pd.read_csv('chatGPT_csvs/chatGPT_perturbedNLP_paraphrase.csv', header=None)

# For each row in stimuli_gpt, split at {243,384}sentences.{number}.
lst_all_sent_candidates = []
for i in range(len(stimuli_gpt)):
    # Get the sentence
    row = stimuli_gpt.iloc[i,0]
    # Split at {243,384}sentences.{number}.
    if '243sentences.' in row:
        row_split = row.split('243sentences.')
    elif '384sentences.' in row:
        row_split = row.split('384sentences.')
    else:
        raise ValueError('Sentence not found in row.')

    row_split = [x for x in row_split if x != '']

    # Get the stimulus_id ({243,384}sentences.{number}) which is the first part of the row
    stimulus_id = row.split(' ')[0].split('.')[:-1]
    stimulus_id = '.'.join(stimulus_id)

    # Check length of row_split (should be 3 or 5). If 5, drop the last 2 elements (we only consider 3 paraphrases per sentence)
    if len(row_split) == 5:
        row_split = row_split[:-2]
    elif len(row_split) != 3:
        raise ValueError('Row split length is not 3 or 5.')

    # Get the sentences (split by space for each item in row_split, and the first element is e.g., "0.1" which means it is sentence_num 0 and candidate 1
    lst_sent_candidates = []
    for sent_candidate in row_split:
        sentence_num = sent_candidate.split(' ')[0].split('.')[0]
        candidate_num = sent_candidate.split(' ')[0].split('.')[1]
        sentence = ' '.join(sent_candidate.split(' ')[1:])
        # Strip whitespace on left and right
        sentence = sentence.strip()

        df_sent_candidate = pd.DataFrame({'stimulus_id': [stimulus_id],
                                'sentence_num': [sentence_num],
                                'candidate_num': [candidate_num],
                                f'sentence_paraphrase_{candidate_num}': [sentence]},
                                index=[candidate_num])
        lst_sent_candidates.append(df_sent_candidate)

        # Assert that last element in stimulus_id is the same as stimulus_id_num
        assert stimulus_id.split('.')[-1] == sentence_num

    # Assert three candidates per sentence
    assert len(lst_sent_candidates) == 3

    # Concatenate the list of dataframes into one dataframe
    # Assert that the stimulus_id and sentence_num is the same for all three candidates
    df_sent_candidates = pd.concat(lst_sent_candidates, axis=0)
    assert len(df_sent_candidates['stimulus_id'].unique()) == 1
    assert len(df_sent_candidates['sentence_num'].unique()) == 1

    # Make into one row, with cols stimulus_id, sentence_num, sentence_paraphrase_1, sentence_paraphrase_2, sentence_paraphrase_3
    df_one_row = pd.DataFrame({'stimulus_id': [stimulus_id],
                                'sentence_num': [sentence_num],
                                'sentence_paraphrase_1': [df_sent_candidates.loc['1', 'sentence_paraphrase_1']],
                                'sentence_paraphrase_2': [df_sent_candidates.loc['2', 'sentence_paraphrase_2']],
                                'sentence_paraphrase_3': [df_sent_candidates.loc['3', 'sentence_paraphrase_3']]},
                                index=[stimulus_id])


    lst_all_sent_candidates.append(df_one_row)


# Concatenate all the dataframes into one dataframe
df_all_sent_candidates = pd.concat(lst_all_sent_candidates)
df_all_sent_candidates['sentence_num'] = df_all_sent_candidates['sentence_num'].astype(int)

### Assertions to ensure that these materials are the same as the original materials
assert len(df_all_sent_candidates) == len(stimuli_df_copy)
assert df_all_sent_candidates['stimulus_id'].tolist() == stimuli_df_copy['stimulus_id'].tolist()
assert df_all_sent_candidates['sentence_num'].tolist() == stimuli_df_copy['sentence_num'].tolist()

# Save
# df_all_sent_candidates.to_csv(f'chatGPT_perturbedNLP_paraphrase_compiled.csv', index=False)

#### Merge with the original stimuli_df_copy
df_merged = pd.merge(stimuli_df_copy, df_all_sent_candidates, on=['stimulus_id', 'sentence_num'], how='left')

# Rename sentence to sentence_original
df_merged = df_merged.rename(columns={'sentence': 'sentence_original'})

# For the sentence, sentence_paraphrase_1, sentence_paraphrase_2, sentence_paraphrase_3, strip punctuation and lowercase. Save in new columns
df_merged['sentence_original_stripped'] = [re.sub(r'[^\w\d\s\'\-\$\%]+', '', sent.lower()) + "." for sent in df_merged['sentence_original']]
df_merged['sentence_paraphrase_1_stripped'] = [re.sub(r'[^\w\d\s\'\-\$\%]+', '', sent.lower()) + "." for sent in df_merged['sentence_paraphrase_1']]
df_merged['sentence_paraphrase_2_stripped'] = [re.sub(r'[^\w\d\s\'\-\$\%]+', '', sent.lower()) + "." for sent in df_merged['sentence_paraphrase_2']]
df_merged['sentence_paraphrase_3_stripped'] = [re.sub(r'[^\w\d\s\'\-\$\%]+', '', sent.lower()) + "." for sent in df_merged['sentence_paraphrase_3']]

# Get sentence lengths
df_merged['sentence_original_length'] = [len(sent.split(' ')) for sent in df_merged['sentence_original_stripped']]
df_merged['sentence_paraphrase_1_length'] = [len(sent.split(' ')) for sent in df_merged['sentence_paraphrase_1_stripped']]
df_merged['sentence_paraphrase_2_length'] = [len(sent.split(' ')) for sent in df_merged['sentence_paraphrase_2_stripped']]
df_merged['sentence_paraphrase_3_length'] = [len(sent.split(' ')) for sent in df_merged['sentence_paraphrase_3_stripped']]

# Find the paraphrased sentence that is closest to the original sentence
for i, row in df_merged.iterrows():
    sent_original = row['sentence_original_stripped']
    sent_paraphrase_1 = row['sentence_paraphrase_1_stripped']
    sent_paraphrase_2 = row['sentence_paraphrase_2_stripped']
    sent_paraphrase_3 = row['sentence_paraphrase_3_stripped']

    # Get the sentence length
    sent_original_length = row['sentence_original_length']
    sent_paraphrase_1_length = row['sentence_paraphrase_1_length']
    sent_paraphrase_2_length = row['sentence_paraphrase_2_length']
    sent_paraphrase_3_length = row['sentence_paraphrase_3_length']

    # Find the sentence that has length closest to the original sentence
    # Get difference in length between original sentence and each paraphrase
    d = {}
    d['1'] = abs(sent_original_length - sent_paraphrase_1_length)
    d['2'] = abs(sent_original_length - sent_paraphrase_2_length)
    d['3'] = abs(sent_original_length - sent_paraphrase_3_length)

    # Get the key with the minimum value
    min_key = min(d, key=d.get)

    # Create a new column with the sentence that is closest to the original sentence
    df_merged.loc[i, 'sentence_paraphrase_closest'] = df_merged.loc[i, f'sentence_paraphrase_{min_key}_stripped']
    df_merged.loc[i, 'sentence_paraphrase_closest_candidate_num'] = min_key
    df_merged.loc[i, 'diff_sentence_length_original_paraphrase'] = int(d[min_key])


# Create a copy of the 'sentence_paraphrase_closest' column (for manual editing) --> name it 'sentence_paraphrase_post_inspection'
df_merged['sentence_paraphrase_post_inspection'] = df_merged['sentence_paraphrase_closest']


# Save the merged dataframe
df_merged.to_csv(f'/Users/gt/Desktop/chatGPT_perturbedNLP_paraphrase_compiled_for-manual-selection.csv', index=False)


print('Number of sentences in the original stimuli set: ', len(stimuli_df_copy))
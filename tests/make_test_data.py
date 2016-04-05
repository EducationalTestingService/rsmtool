
import argparse
import logging

import pandas as pd
import numpy as np

def main():

    # set up an argument parser
    parser = argparse.ArgumentParser(prog='make_test_data.py')
    parser.add_argument('infile', help="The ASAP2 CSV file with feature values")
    parser.add_argument('--prompts',
                        dest='num_prompts',
                        type=int,
                        help="The number of additional fake prompts to create",
                        required=True)
    parser.add_argument('--rows',
                        dest='num_rows',
                        type=int,
                        help="The number of rows for each prompt",
                        required=True)
    parser.add_argument('--out',
                        dest='output_file',
                        help="The output CSV file",
                        required=True)


    # parse given command line arguments
    args = parser.parse_args()

    # set up the logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    # read in the given file into a data frame
    df = pd.read_csv(args.infile)

    # randomly sample 100 rows from this data frame
    prng = np.random.RandomState(1234)
    random_rows = prng.choice(df.index.values, size=args.num_rows, replace=False)
    df_chosen = df.iloc[random_rows].copy()
    df_chosen['QUESTION'] = 'QUESTION_1'

    # get a list of the feature columns
    feature_columns = [c for c in df_chosen.columns if c.startswith('FEATURE')]

    # now we need to generate 100 rows for each new prompt we are adding
    # and just jitter the existing feature values in each row in the
    # original data frame. Also add the new question field.
    new_prompt_dfs = []
    for prompt_num in range(args.num_prompts):
        df_new = df_chosen.copy()
        for column in feature_columns:
            feature_values = df_chosen[column].values
            scale_values = feature_values / 100
            scale_values[scale_values == 0] = 0.001
            df_new[column] = prng.normal(loc=feature_values,
                                         scale=np.abs(scale_values))
        df_new['QUESTION'] = 'QUESTION_{}'.format(prompt_num + 2)
        new_prompt_dfs.append(df_new)

    # concatenate all the data frames together
    df_all_new = pd.concat([df_chosen] + new_prompt_dfs)
    df_all_new.reset_index(drop=True, inplace=True)

    # now we want to add two new randomly generated features
    # 1. FEATURE7: integer values between 1 and 20
    # 2. FEATURE8: very small negative values (-12000 to -5000) which
    #              does not look like a z-score

    df_all_new['FEATURE7'] = prng.random_integers(1, 20, len(df_all_new))
    df_all_new['FEATURE8'] = 1000 * (prng.random_integers(-12, -5, len(df_all_new)) + \
                                     prng.normal(0, 1, len(df_all_new)))

    # add a fake L1 to each of the rows in the data frame
    df_all_new['L1'] = prng.choice(['Klingon', 'Esperanto', 'Navi', 'Vulcan'], len(df_all_new))

    # reorder the columns
    df_all_new = df_all_new[['ID'] + feature_columns + ['FEATURE7', 'FEATURE8', 'LENGTH', 'QUESTION', 'L1', 'score', 'score2']]

    # reset the IDs to be unique
    df_all_new['ID'] = ['RESPONSE_{}'.format(i) for i in range(1, len(df_all_new) + 1)]

    # write out the data frame to the output file
    df_all_new.to_csv(args.output_file, index=False)

if __name__ == '__main__':
    main()

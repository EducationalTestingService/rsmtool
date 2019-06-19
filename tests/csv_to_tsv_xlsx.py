# python

'''
A utility script to convert csv files to tsv and xlsx'''

import pandas as pd
from os.path import splitext

FILELIST = ['data/files/train.csv',
            'data/files/test.csv',
            'data/files/predictions_raw.csv',
            'data/experiments/lr-with-feature-subset-file/feature_file.csv',
            'data/files/train_predictions.csv'
            ]

def main():
    for f in FILELIST:
        d = pd.read_csv(f)
        tsv_name = splitext(f)[0]+'.tsv'
        xlsx_name = splitext(f)[0]+'.xlsx'
        print(tsv_name, xlsx_name)
        d.to_csv(tsv_name, sep='\t', index=False)
        d.to_excel(xlsx_name, sheet_name='data', index=False)



if __name__ == '__main__':
    main()

"""
Utility classes and functions related to computing fairness evaluations

:author: Anastassia Loukina (aloukina@ets.org)
:author: Jeremy Biggs (jbiggs@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:date: 07/29/2019
:organization: ETS
"""


import json
import pandas as pd
import os
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np
from statsmodels.stats.anova import anova_lm
from matplotlib import pyplot as plt

def get_equal_sample(df):
    
    '''Generate sample with the same score distributions
    for each Native Language as
    described in the paper'''
    
    counts = pd.crosstab(df['sc1'], df['NativeLang'])
    to_sample = counts.apply(min, axis=1) 
    def sample_data(df_lang, to_sample):
        n = to_sample[df_lang['sc1'].values[0]]
        df_out = df_lang.sample(n, random_state=20000)
        return(df_out)

    df_equal = df.groupby(['sc1', 'NativeLang']).apply(sample_data, to_sample).reset_index(drop=True)
    return df_equal


def get_significant(res):
        out_dict = {}
        sigs = list(res.pvalues[res.pvalues<0.01].index)
        sig_langs = [v for v in sigs if not v in ['sc1',
                                                  'raw_trim',
                                                  'group_sc1[T.1.0]',
                                                  'group_sc1[T.2.0]',
                                                  'group_sc1[T.3.0]',
                                                  'group_sc1[T.4.0]']]
        for v in sig_langs:
            if v == 'Intercept':
                key = v
            else:
                key = v.split('.')[1].strip(']')
            out_dict[key] = res.params[v]
        return out_dict


def get_fairness_analysis(df, group,
                          system_score_column='raw_trim',
                          human_score_column='sc1'):
    ''' Main function for computing various fairness metrics'''
    
    # compute error and squared error

    df['error'] = df['raw_trim']-df['sc1']
    df['SE'] = df['error']**2

    # convert group values to category and reorder them using 
    # the largest category as reference

    df['group'] = df[group].astype("category")
    df['group'] = df['group'].cat.reorder_categories(['SPA', 'ARA', 'CHI', 'GER', 'JPN', 'KOR'], ordered=True)

    
    # Overall score accuracy (OSA)
    # Variance in squared error explained by L1
    
    mod1 = smf.ols(formula='SE ~ group_nl', data=df)
    res1 = mod1.fit()


    df_1 = {'R2': res1.rsquared_adj}
    df_1.update(get_significant(res1))
    df_1['sig'] = np.round(res1.f_pvalue, 5)
    df_1 = pd.Series(df_1, name='Overall squared error')


    # Overall score difference (OSD)
    # variance in signed residuals (raw error) explained by L1


    df['group_sc1'] = df['sc1'].astype("category")
    df['group_sc1'] = df['group_sc1'].cat.reorder_categories([3.0, 1.0, 2.0, 4.0], ordered=True)
    mod1 = smf.ols(formula='error ~ group_nl', data=df)
    res1 = mod1.fit()


    df_1 = {'R2': res1.rsquared_adj}
    df_1.update(get_significant(res1))
    df_1['sig'] = np.round(res1.f_pvalue, 5)
    df_1 = pd.Series(df_1, name='Overall error')

    # conditional score difference CSD 
    # Variance in score difference conditioned on Native language
    
    mod0 = smf.ols(formula='error ~ group_sc1', data=df)
    res0 = mod0.fit()
    mod2 = smf.ols(formula='error ~ group_nl+group_sc1', data=df)
    res2 = mod2.fit()


    df_2 = {'R2': res2.rsquared_adj - res0.rsquared_adj}
    df_2.update(get_significant(res2))
    df_2['sig'] = np.round(anova_lm(res0, res2).values[1][-1], 5)
    df_2 = pd.Series(df_2, name='Conditional procedure error')


    df_all = pd.concat([df_3, df_1, df_2], axis=1, sort=True)
    return df_all


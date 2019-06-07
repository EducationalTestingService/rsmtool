"""
Utility classes and functions related to computing test
theory based evaluations

:author: Anastassia Loukina (aloukina@ets.org)
:author: Jeremy Biggs (jbiggs@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:date: 05/20/2019
:organization: ETS
"""

import pandas as pd



def compute_variance_of_errors(df):
    """ Compute variance of errors in human scores

    Parameters
    ----------
    df: pandas DataFrame
        Input dataframe. Must contain columns 'sc1' and 'sc2'
     
    Returns
    -------
    v_e: float
         Variance of errors in human scores

    Raises
    ------
    ValueError
         If some of the values in `sc1` or `sc2` are None.
    """

    # check that all responses are double scored
    if (len(df[df['sc2'].isnull()]) > 0) or (len(df[df['sc1'].isnull()]) > 0):
        raise ValueError("Variance of errors should only be computed on double-scored responses")

    N = len(df)

    v_e = 1/(2*N) * ((df['sc1'] - df['sc2'])**2).sum()

    return v_e


def compute_true_score_var_subset_double_scored(single_human_scores,
                                                double_human_scores,
                                                v_e):
    """ Compute variance of true scores
    in a situation where only some
    of the responses are double scored

    Parameters
    ----------
    single_human_scores: pandas Series
        Human scores for single-scored responses      
    double_human_scores: pandas Series
       Human scores for double-scored responses         
    v_e: float
        Variance of errors in human scores

    Returns
    -------
    v_t: float
        variance of true scores
        """

    # N of examinees with 1 human score.
    n_1 = len(single_human_scores)

    # N of examinees with 2 human scores.
    n_2 = len(double_human_scores)

    N = n_1 + n_2

    # compute squared distance c_i*(sc_bar_i - sc_bar)**2 for each response.  Note that for double scored responses
    # c=2.

    sc_bar_mean = pd.concat([single_human_scores, double_human_scores]).mean()

    squared_dist_single = (single_human_scores - sc_bar_mean)**2
    squared_dist_double = 2 * (double_human_scores - sc_bar_mean)**2

    # concatenate both dataframes
    squared_dist = pd.concat([squared_dist_single, squared_dist_double], sort=True)

    # third, compute variance of true scores
    numerator = (squared_dist.sum() - (N-1) * v_e)

    denominator = (N-1) + (n_2 * (n_1 + 2*n_2 - 2))/(n_1 + 2 * n_2)

    var_t = numerator/denominator

    return var_t



def compute_mse_subset_double_scored(single_human_scores,
                                     double_human_scores,
                                     single_system_scores,
                                     double_system_scores,
                                     v_e):
    """ Compute MSE for predicting true score from system scores
    in a situation where only some
    of the responses are double scored

    Parameters
    ----------
    single_human_scores: pandas Series
        Human scores for single-scored responses      
    double_human_scores: pandas Series
       Human scores for double-scored responses   
    single_system_scores: pandas Series
    	System scores for single_scored responses
    double_human_scores: pandas Series
    	System scores for double_scored responses      
    v_e: float
        Variance of errors in human scores


    Returns
    -------
    mse: float
        mse
        """

    # N of examinees with 1 human score.
    n_1 = len(single_human_scores)

    # N of examinees with 2 human scores.
    n_2 = len(double_human_scores)

    N = n_1 + n_2

    # compute squared error c_i*(sc_bar_i - system_i)**2 . 
    # Note that for double-scored responses c= 2. 
    se_single = (single_human_scores - single_system_scores)**2
    se_double = 2 * (double_human_scores - double_system_scores)**2

    # concatenate both sets
    se = pd.concat([se_single, se_double], sort=True)

    # Compute mean squared error for predicting true score
    mse = (se.sum() - N*v_e) / (n_1 + 2*n_2)

    print(mse)
    return mse


def compute_true_score_var_all_double_scored(human_scores,
											 v_e):
    """ Compute variance of true scores
    in a situation where only some
    of the responses are double scored

    Parameters
    ----------
    human_scores: pandas Series
        Human scores used to compute the variance     
    v_e: float
        Variance of errors in human scores

    Returns
    -------
    v_t: float
        variance of true scores"""

    N = len(human_scores)

    var_t = ((human_scores - human_scores.mean())**2).sum() / (N-1) - v_e/2

    return var_t


def compute_mse_all_double_scored(human_scores,
								  system_scores,
                              	  v_e):

    """ Compute MSE for predicting true score from system scores
    in a situation where all
    of the responses are double scored

    Parameters
    ----------
    human_scores: pandas Series
        Human scores   
	system_scores: pandas Series
    	System scores   
    v_e: float
        Variance of errors in human scores
    Returns
    -------
    mse: float
        mse"""

    N = len(human_scores)
    
    # compute mse_t_m
    mse = ((human_scores - system_scores)**2).sum()/N - v_e/2

    return mse


def compute_prmse(df, system_score_columns):
    """ Compute Proportional Reduction in Mean Squared Error (PRMSE) 
    for predicting true score from system scores.

    Parameters
    ----------
    df: pandas DataFrame
        Input dataframe. Must contain columns `sc1`, `sc2` and the columns 
        `listed in system_score_columns`.   
    system_score_columns: list
        List of columns containing system scores

    Returns
    -------
    prmse_metrics: pandas Series
        Series containing different evaluation metrics related to the evaluation
        of system scores against true scores:
   
        - `N`: total number of responses
        - `N_single`: total number of responses with a single human score
        - `N_double`: total number of responses with two human scores
        - `h1_var_single`: variance of first human score for single-scored responses
        - `h1_var_double`: variance of first human score for double-scored responses
        - `h2_var_double`: variance of second human score for double-scored responses
        - `tru_var`: estimated true score variance
        - `system_var_all`:  variance of system scores for all responses
        - `system_var_double`:  variance of system scores for double-scored responses
        - `mse_true`: mean squared error for predicting true score from machine score
        - `prmse`: proportional reduction in mean squared error for predicting true score

            """

    score_mask = df['sc2'].isnull() 

    df_single = df[score_mask].copy()
    df_double = df[~score_mask].copy()

    # compute variance of errors
    v_e = compute_variance_of_errors(df_double)

    # compute average score for double-scored responses
    df_double['sc_bar'] = (df_double['sc1'] + df_double['sc2'])/2

    # compute variance of true scores

    if len(df_single) > 0:     
        # compute 
        var_t = compute_true_score_var_subset_double_scored(df_single['sc1'],
                                                            df_double['sc_bar'],
                                                            v_e)

    else:
        var_t = compute_true_score_var_all_double_scored(df_double['sc_bar'],
                                                         v_e)
        
    # compute MSE for each type of score
    prmse_all = []
    for system in system_score_columns:
        if len(df_single) > 0:
            mse = compute_mse_subset_double_scored(df_single['sc1'],
            									   df_double['sc_bar'],
            									   df_single[system],
            									   df_double[system],
                                                   v_e)
        else:    
            mse = compute_mse_all_double_scored(df_double['sc_bar'],
            									df_double[system],
                                                v_e,)

    
        prmse_metrics = pd.Series({'sys_var_single': df_single[system].var(ddof=1),
                                   'sys_var_double': df_double[system].var(ddof=1),
                                   'mse_true': mse,
                                   'prmse_true': 1-mse/var_t}, name=system)
        prmse_all.append(prmse_metrics)
        
    # combine all results together
    df_prmse = pd.concat(prmse_all, axis=1, sort=True).transpose()
    
    # add numbers that are the same for all types of scores
    df_prmse.insert(0, 'N', len(df))
    df_prmse.insert(1, 'N_single', len(df_single))
    df_prmse.insert(2, 'N_double', len(df_double))
    df_prmse.insert(3, 'h1_var_single', df_single['sc1'].var(ddof=1))
    df_prmse.insert(4, 'h1_var_double', df_double['sc1'].var(ddof=1))
    df_prmse.insert(5, 'h2_var_double', df_double['sc2'].var(ddof=1))
    df_prmse.insert(6, 'true_var', var_t)
    
    return df_prmse


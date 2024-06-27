import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from matplotlib.ticker import ScalarFormatter as mpl_ScalarFormatter
import yaml
with open("wdir.yml",
          "r",
          encoding="utf8") as file_config:
    config = yaml.safe_load(file_config)

directory = f"{config['PATH']}{config['dir_out']}"


condition_labels = pd.read_excel(
    f'{config["PATH"]}{config["dir_data"]}aurumGoldLabels_NCQOF.xlsx',
    header=3).dropna(subset=['Joht Labelling']) #condition labels file should also be available

os.chdir(directory)
data_root = "./"

missing_files = ['CPRDNHL4']
condition_labels = condition_labels[~condition_labels['GOLD'].isin(missing_files)] #remove missing codes

#GOLD_labels = condition_labels['GOLD'] #not needed here
JOHT_labels = condition_labels['Joht Labelling'] #instead of running through all unique conditions, analysis will run through JOHT labelled conditions, throwing up errors for missing values

#standardised_data_root = 'Standardised Results 10-05'

group_col = 'Group'
category_col = 'Subgroup'
condition_col = 'Condition'
date_col = 'Date'
col_variables = [condition_col,
                 category_col,
                 group_col] #order important, change names only
include_groups = ['Ethnicity','Region','Deprivation','Overall'] #drop pairwise combinations
#exclude_groups = ['RegionEth','DeprivationEth']

analysis_dates = ['2001-01-01', '2002-01-01', '2003-01-01', '2004-01-01',
       '2005-01-01', '2006-01-01', '2007-01-01', '2008-01-01',
       '2009-01-01', '2010-01-01', '2011-01-01', '2012-01-01',
       '2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01',
       '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01',
       '2021-01-01']

prev_analysis_dates = ['2001-07-01', '2002-07-01', '2003-07-01', '2004-07-01',
       '2005-07-01', '2006-07-01', '2007-07-01', '2008-07-01',
       '2009-07-01', '2010-07-01', '2011-07-01', '2012-07-01',
       '2013-07-01', '2014-07-01', '2015-07-01', '2016-07-01',
       '2017-07-01', '2018-07-01', '2019-07-01', '2020-07-01',
       '2021-07-01']

years = ['2001', '2002', '2003', '2004',
       '2005', '2006', '2007', '2008',
       '2009', '2010', '2011', '2012',
       '2013', '2014', '2015', '2016',
       '2017', '2018', '2019', '2020',
       '2021']

measure = 'Prevalence'
prev1 = pd.read_csv(data_root + 'prev_DSR.csv') #note expected column structure as below, with names assigned as above
#prev1.columns = [condition_col,group_col,date_col,category_col,'LowerCI',measure,'UpperCI']
prev1 = prev1.rename(columns={
    'Condition': condition_col,
    'Subgroup': group_col,
    'Year': date_col,
    'Group': category_col,
    'LowerCI': 'LowerCI',
    measure: measure,
    'UpperCI': 'UpperCI'
    })
prev1 = prev1[prev1[date_col].isin(prev_analysis_dates)] #only include observations from years in the analysis_dates
prev1 = prev1[prev1[category_col].isin(include_groups)] #"include groups" only
prev1 = prev1[col_variables+[date_col,measure]] #drop CIs
prev1 = prev1.set_index(col_variables+[date_col]).unstack(3) #put date_col in wide format
prev1.columns = analysis_dates

measure = 'Incidence'
inc1 = pd.read_csv(data_root + 'inc_DSR.csv')
inc1 = inc1.rename(columns={
    'Condition': condition_col,
    'Subgroup': group_col,
    'Year': date_col,
    'Group': category_col,
    'LowerCI': 'LowerCI',
    measure: measure,
    'UpperCI': 'UpperCI'
    })
#inc1.columns = [condition_col,group_col,date_col,category_col,'LowerCI',measure,'UpperCI']
inc1 = inc1[inc1[date_col].isin(analysis_dates)] #only include observations from years in the analysis_dates
inc1 = inc1[inc1[category_col].isin(include_groups)] #"include groups" only
inc1 = inc1[col_variables+[date_col,measure]]
inc1 = inc1.set_index(col_variables+[date_col]).unstack(3)


idx = pd.IndexSlice

#every condition-measure-year-subgroup has a point estimate. We need a ratio of this to the overall point estimate for that condition-measure-year.
def geom_mean(ratio_data,ratio_col):
    """return the geometric mean of a dataset. Note 0 values are dropped
    geom_mean(a) = exp(arith_mean(ln(a)))
    input: series or dataframe
    output: geometric mean of series or series of geometric means for dataframe cols
    """
    ratio_data = ratio_data.replace(0,np.NaN)
    ratio_data = ratio_data[ratio_col]
    if isinstance(ratio_data,pd.core.series.Series):
        return np.exp(np.log(ratio_data).sum()/ratio_data.notna().sum())

    elif isinstance(ratio_data,pd.core.frame.DataFrame):
        return np.exp(np.log(ratio_data.prod(axis=1))/ratio_data.notna().sum(1))

    else:
        raise warning('unrecognised data type provided to geometric mean function')

def calculate_overall_ratios(df,measure,condition,ratio_name):
    """ Calculate a ratio of a subgroup condition-year-estimate to the overall condition-year-estimate

        inputs
                df: a wideform dataframe of all subgroup results for a measure-and-condition, col per date
                measure: a string of either Incidence or Prevalence
                condition: a string of the condition name
                ratio_name: a string of 'ratio' + measure

        outputs
                a dataframe of a single condition with float cols for estimate and for ratio for all years.
                Multiindex of condition(?), category, subgroup. Multiindex cols of measure vs ratio, dates"""
    df_condition_measure = df.loc[idx[condition],:] #a df of all point-estimates for a measure-and-condition, one row per category, many for dates
    #display(df_condition_measure.head())
    df_condition_ratio = df_condition_measure / df_condition_measure.loc[idx['Overall',:]].values
    #df_condition_measure.loc[idx[:,'Overall',:]].values returns the floats for categories (not subgroups) named 'Overall' only

    #ratio_name = measure + ' Ratio'

    return pd.concat([df_condition_measure,df_condition_ratio],
                     axis=1,
                     keys=[measure,ratio_name])

def calculate_measure_ratios(df,measure,ratio_name):
    """ return ratios for all conditions for one measure
        ouputs
        dataframe of col_variables + ['Date',measure, ratio_name], single level index. Long format, single date per row"""
    ratios = {}
    for i in JOHT_labels:
        # calculate overall ratios
        ratios[i] = calculate_overall_ratios(df,measure,i,ratio_name)
    #ratio_name = measure + ' Ratio'
    df1 = pd.concat(ratios,
              axis=1).stack().stack(level=0).reset_index()
    df1 = df1[[df1.columns[3]] + list(df1.columns[:3]) + list(df1.columns[4:])]
    df1.columns = col_variables + ['Date',measure, ratio_name]
    return df1

def calculate_all_ratios(df, i, mean='Geometric'):
    """ calculate ratios for a dataframe of all results for a measure i

        return a dataframe of all conditions, dates, subgroups for a measure, with a col for estimate and for ratio"""
    df_measure = df.reset_index()
    df_measure.columns = col_variables + analysis_dates
    df_measure = df_measure.set_index(col_variables)

    ratio_name = i + ' Ratio'


    df = calculate_measure_ratios(df,
                                    i,
                                    ratio_name)

    if mean=='Arithmetic':
        return  (df, #measure, ratio results dict
                df.groupby([group_col,'Date']).mean()[ratio_name].unstack(),
                 #mean ratios for subgroups by year. Lookup for calculating expecteds later
                df.groupby(group_col).mean()[ratio_name])
                #mean ratios for all condition-year-subgroups. Lookup for calculating expecteds later

    elif mean == 'Geometric':
        return  (df, #measure, ratio results dict
                df.groupby([group_col,'Date']).apply(geom_mean,ratio_col=ratio_name).unstack(),#mean ratios for subgroups by year. Lookup for calculating expecteds later
                df.groupby(group_col).apply(geom_mean,ratio_col=ratio_name))#[ratio_name]) #mean condition-subgroup ratios for all years. Lookup for calculating expecteds later

####################################################################

measure_dict = {'Incidence':inc1,
                'Prevalence':prev1} #dataframes of standardised results, plus name (for columns later)
#these are reading in as a multindex. detect multiindex and drop top level if present:
for i in measure_dict:
    if isinstance(measure_dict[i].columns, pd.MultiIndex):
        measure_dict[i].columns = measure_dict[i].columns.droplevel()

expected_ratio_yearly_dict = {}
expected_ratio_overall_dict = {}


for i in measure_dict:
    (measure_dict[i], #measure, ratio results dict
     expected_ratio_yearly_dict[i], #mean ratios for subgroups by year. Lookup for calculating expecteds later
     expected_ratio_overall_dict[i]) = calculate_all_ratios(measure_dict[i], i, mean='Geometric')

df2 = pd.merge(measure_dict['Prevalence'],measure_dict['Incidence'],how='outer')

yearly_ratios = pd.concat(expected_ratio_yearly_dict)

def df_of_expected_prev_condition_year_spec(df,ratio,measure,date):
    """ Calculate subgroup expected estimate for all conditions from overall condition estimate x subgroup mean ratio

    outputs
    dataframe for one year of index [condition, measure], col of expected estimate"""
    mask1 = df[group_col]=='Overall'
    mask2 = df[date_col]==date

    dfx = df[mask1&mask2]
    dfx = dfx[[condition_col,measure]]
    dfx = dfx.set_index(condition_col)*ratio
    return dfx

def find_ratio(group,date,measure,lookup = expected_ratio_yearly_dict):
    """ return the ratio for a subgroup for a year
    outputs a float"""
    for i in lookup:
        lookup[i].columns= analysis_dates

    return lookup[measure].loc[group,date]

def group_expecteds_both_measure(df,group,date):
    """ Return expected estimates for all conditions and both measures for a subgroup-date pair.
    Adds col multindex layer for measure"""
    dict_expected_individual_measure = []

    for measure in measure_dict.keys():
        group_ratio = find_ratio(group,date,measure)
        dict_expected_individual_measure.append(
            df_of_expected_prev_condition_year_spec(df,group_ratio,measure,date)
            )

    return pd.concat(dict_expected_individual_measure,axis=1)

def expected_all_groups_dates(df):
    """ Return expected estimates for both measure, all conditions, dates, subgroups

    Returns a long format df with one condition-date-subgroup per row, expected measures for both"""
    dict_expected_group_dates = {}

    for group in df[group_col].unique():
        dict_expected_dates = {}

        for date in analysis_dates:
            dict_expected_dates[date] = group_expecteds_both_measure(df,group,date)
        dict_expected_group_dates[group] = pd.concat(dict_expected_dates)
    dfx = pd.concat(dict_expected_group_dates)
    dfx = dfx.reset_index()
    dfx.columns = [group_col,date_col,condition_col,
                   'Expected Incidence','Expected Prevalence'] #hard coded :'(
    return dfx

df3 = expected_all_groups_dates(df2)

index_cols = [condition_col,date_col,group_col]
df4 = pd.merge(df2.set_index(index_cols),
               df3.set_index(index_cols),
               left_on=index_cols,
               right_on=index_cols)

def lower_observed_z(o,e): #subject to change based on negative binomial formulas
    """ return z-score (standard deviations representation of p-value)"""
    p = ss.chi2.sf(x=o*2,df=e*2) #subject to change
    return ss.norm.ppf(p)*-1

def upper_observed_z(o,e):#subject to change based on negative binomial formulas
    """ return z-score (standard deviations representation of p-value)"""
    p = ss.chi2.sf(x=o*2,df=2*(e+1)) #subject to change
    return ss.norm.ppf(p)*-1

def lower_wrapper(dfx, o,e):
    """ vectorise z-score calculation"""
    return lower_observed_z(dfx[o], dfx[e])

def upper_wrapper(dfx, o,e):
    """ vectorise z-score calculation"""
    return upper_observed_z(dfx[o], dfx[e])

def calculate_subgroup_z_scores(df,o,e):
    """Inputs:
          dataframe with observed and expected values,
          string name of observed column,
          string name of expected column
    Outputs:
          df with z column"""
    z_scores = pd.concat([df[df[o]<=df[e]].apply(lower_wrapper,axis=1,args=(o,e)),
                          df[df[o]>df[e]].apply(upper_wrapper,axis=1,args=(o,e))])
    z_scores.name = o + ' Z-Score'
    return z_scores.sort_index()

#breakpoint()
#Prevalence should be per 1,000 for the z-scores, as the arbitrarily larger rate inflates the z-score when denominator is 100,000.
#Maximum rate should be 3 figures to get roughly consistent scale for standardisation.
#At this point prevalence is per 100,000, so divide by 100
print('Prevalence is assumed to be per 100,000 and is divided by 100 for z-scoring')
df4.Prevalence = df4.Prevalence/100
df4['Expected Prevalence'] = df4['Expected Prevalence'] /100

prev_z = calculate_subgroup_z_scores(df4.drop('Overall',level='Group'),'Prevalence','Expected Prevalence')
inc_z = calculate_subgroup_z_scores(df4.drop('Overall',level='Group'),'Incidence','Expected Incidence')

df5 = pd.concat([df4,#.drop('Overall',level='Group'),
          prev_z,
          inc_z],axis=1).reset_index()

df5 = df5[['Condition', 'Date', 'Group', #'Category',
           'Prevalence','Prevalence Ratio',  'Expected Prevalence',  'Prevalence Z-Score',
           'Incidence', 'Incidence Ratio', 'Expected Incidence',  'Incidence Z-Score']]

if not os.path.exists("Average_Geometric"):
    os.mkdir("Average_Geometric")

df5.to_csv(f"Average_Geometric/281 conditions chi2 z-scores, expected and observed rates.csv")
pd.concat(expected_ratio_yearly_dict).to_csv(f"Average_Geometric/281 conditions yearly subgroup-overall ratios.csv")
pd.concat(expected_ratio_overall_dict).to_csv(f"Average_Geometric/281 conditions 20-year subgroup-overall ratios.csv")

#df5.groupby("Group").mean()
#df5[(df5["Group"]=="'BLACK'")]["Prevalence Ratio"].describe()


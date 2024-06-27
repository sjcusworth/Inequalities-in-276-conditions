import pandas as pd
import os

cwd = '/rds/projects/c/chandjsz-analogy-automatedanalytics/281Cond/'
os.chdir(cwd)

cols = ['SEX',
           'ETHNICITY',
       'HEALTH_AUTH',
       'AGE_CATEGORY',
       'IMD_pracid',
       'YEAR_OF_BIRTH',
        'START_DATE',
       'EXIT_DATE']
df = pd.read_parquet('data/dat_processed.parquet',
                    columns=cols)

cats=['SEX',
           'ETHNICITY',
       'HEALTH_AUTH',
       'AGE_CATEGORY',
       'IMD_pracid']
table1 = {}
for i in cats:
    table1[i] = df[i].value_counts()

mask_start = pd.to_datetime(df['START_DATE']) < pd.Timestamp('2019-01-01')
mask_end = pd.to_datetime(df['EXIT_DATE']) > pd.Timestamp('2020-01-01')
age2019 = 2019 - df.YEAR_OF_BIRTH.str[:4].astype(int) 

df['Age2019'] = pd.cut(age2019,[0,16,30,40,50,60,70,80,115],
                       labels= ['0-16', '17-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81+'])

cats=['SEX',
           'ETHNICITY',
       'HEALTH_AUTH',
       'Age2019',
       'IMD_pracid']
table2019 = {}
for i in cats:
    table2019[i] = df[mask_start&mask_end][i].value_counts()

table2019['AGE_CATEGORY'] = table2019.pop('Age2019')

for dfx in table1, table2019:
    dfx['SEX'] = dfx['SEX'].loc[['F','M','I']]
    dfx['SEX'].index = ['Female','Male','Intersex']
    
    dfx['ETHNICITY'] = dfx['ETHNICITY'].sort_index()
    dfx['HEALTH_AUTH'] = dfx['HEALTH_AUTH'].sort_index()
    
    dfx['IMD_pracid'] = dfx['IMD_pracid'].loc[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Ireland']]
    dfx['IMD_pracid'].index = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Ireland (No IMD available)']

df1 = pd.concat(table1).astype(str) + ' (' + (100*pd.concat(table1)/len(df)).round(1).astype(str) + ')'
df2019 = pd.concat(table2019).astype(str) + ' (' + (100*pd.concat(table2019)/len(df[mask_start&mask_end])).round(1).astype(str) + ')'

df_bc = pd.concat([df1,df2019],
         axis=1,keys=['All included participants at entry to study',
                        'Participant at 2019-01-01']
         ).fillna('0')
new_cols = ['Category',
                 'Subgroup',
    'All included participants at entry to study',
                        'Participants in 2019']

overall = pd.Series(['n','',
            len(df),
           len(df[mask_start&mask_end])],
         index=new_cols).astype(str)
#df_bc.loc[('n',''),:] = overall
df_bc = df_bc.reset_index()
df_bc.columns = new_cols
categories = ['AGE_CATEGORY','SEX', 'ETHNICITY','HEALTH_AUTH','IMD_pracid']
category_names = ['Age Groups, n(%)','Sex, n (%)','Ethnicity, n (%)','Region, n (%)','IMD, n (%)']
cat_dict = dict(zip(categories,category_names))
df_bc.Category = df_bc.Category.map(cat_dict)
df_bc.Category = pd.Categorical(df_bc.Category, category_names)
df_bc.Subgroup = df_bc.Subgroup.str.title()
df_bc = df_bc.set_index(['Category','Subgroup'])

df_bc = df_bc.reset_index()
df_bc.index = df_bc.index + 1  # shifting index
df_bc.sort_index(inplace=True) 
df_bc.to_csv('out/Publish/tableOne.csv')
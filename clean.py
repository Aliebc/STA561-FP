## Cleaning CHFS 2017 Individual Data
import pandas as pd
from _tool import (
    load_source_data, 
    save_cleaned_data, 
    filter_columns,
    get_cityid
)
import json

df_ind2017 = load_source_data('chfs2017_ind_202104.dta')
df_ind2017_county = load_source_data('chfs2017_ind_county.dta')
df_income2017 = load_source_data('INCOME2017.dta')

TARGET_COLUMNS = [
    'hhid_2017',
    'pline',
    'hhead',
    'a2001',
    'a2003',
    'a2005',
    'a2006',
    'a2012',
    'a2012a',
    'a2019',
    'a2019b',
    'a2022',
    'a2024',
    'a2025b',
    'a2028',
    'a2029',
    'a3100',
    'a3105',
    'a3106',
    'a3136',
    'a3109',
    'a3110',
    'a3111',
    'a2015',
    'a2013ac',
    'a3118',
    'f1001a',
    'f1005',
    'a2015'
]

df_ind2017 = filter_columns(df_ind2017, TARGET_COLUMNS)

df_ind2017_county = df_ind2017_county[[
    'hhid_2017',
    'pline',
    'countyid',
]]

df_income2017 = df_income2017[[
    'Ctnm_id',
    'Eect13'
]]
df_income2017.rename(columns={'Ctnm_id': 'cityid', 'Eect13': 'cincome'}, inplace=True)

df_ind2017_county['cityid'] = df_ind2017_county['countyid'].apply(get_cityid)

df_ind2017_county = pd.merge(df_ind2017_county, df_income2017, on='cityid', how='left')

df_ind2017 = pd.merge(df_ind2017, df_ind2017_county, on=['hhid_2017', 'pline'], how='left')

df_ind2017['age'] = 2017 - df_ind2017['a2005']

df_ind2017['a3109'] = df_ind2017['a3109'].mask(
    df_ind2017['a3109'].isna(),
    df_ind2017['a3136'] / 12
)
def select_family(df: pd.DataFrame):
    # 户主年龄在25-50岁之间，户主是hhead ==1    
    if df.loc[df['hhead'] == 1, 'age'].between(25, 60).any():
        if df['a2001'].isin([3, 4]).any():
            return True
    # 儿女年龄在25-50岁之间
    if df.loc[df['a2001'] == 6, 'age'].between(25, 60).any():
        return True
    return False

# 筛掉a3109有缺失值的家庭
df_ind2017 = df_ind2017.groupby('hhid_2017').filter(select_family).reset_index(drop=True)

def rename_columns(list_columns, name) -> dict:
    rename_dict = {}
    for column in list_columns:
        rename_dict[column] = f"{column}_{name}"
    return rename_dict
    

def apply_family(df: pd.DataFrame):
    cols = df.columns
    df.drop(columns=['hhead'], inplace=True)
    # 提取每类成员，按家庭编号 hhid_2017 为索引
    hhead = df[(df['a2001'] == 1) & (df['a3109'].notna())].set_index('hhid_2017', drop=True)
    hhead = hhead[hhead['age'].between(25, 50)]
    father = df[(df['a2001'] == 3) & (df['a2003'] == 1)].set_index('hhid_2017', drop=True)
    mother = df[(df['a2001'] == 3) & (df['a2003'] == 2)].set_index('hhid_2017', drop=True)

    # 重命名列加前缀
    hhead = hhead
    father = father.rename(columns=rename_columns(cols, 'father'))
    mother = mother.rename(columns=rename_columns(cols, 'mother'))

    # 以户主为主表，join 父母信息（横向拼接）
    result = hhead.join(father, how='left').join(mother, how='left')

    return result

#df_ind2017 = df_ind2017.groupby('hhid_2017').apply(apply_family)

df_ind2017_1 = df_ind2017[df_ind2017['a2001'].isin([1, 3])]
df_ind2017_1['hhid_2017'] = df_ind2017_1['hhid_2017'].astype(str)

df_ind2017_2 = df_ind2017[df_ind2017['a2001'].isin([2, 4])]
df_ind2017_2['hhid_2017'] = df_ind2017_2['hhid_2017'].astype(str)
df_ind2017_2['a2001'] = df_ind2017_2['a2001'].replace({2: 1, 4: 3})

df_ind2017_1f = apply_family(df_ind2017_1)
df_ind2017_2f = apply_family(df_ind2017_2)
df_ind2017 = pd.concat([df_ind2017_1f, df_ind2017_2f], axis=0)

df_ind2017 = df_ind2017.sort_values(by=['hhid_2017', 'pline'])

print(df_ind2017)


reader = pd.io.stata.StataReader('data/chfs2017_ind_202104.dta')
original_labels = reader.variable_labels()
original_labels['age'] = '年龄'
original_labels['countyid'] = '县级行政区划代码'
original_labels['cityid'] = '市级行政区划代码'
original_labels['cincome'] = '市级收入水平'
new_labels = {}

for k, v in original_labels.items():
    new_labels[k] = v
    new_labels[k + '_father'] = v + '(父亲)'
    new_labels[k + '_mother'] = v + '(母亲)'
    
#json.dump(new_labels, open('clean/var_list.json', 'w'), ensure_ascii=False, indent=4)

save_cleaned_data(df_ind2017, 'chfs2017_income.dta', labels=new_labels)
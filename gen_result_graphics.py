import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def gen_graphics(df,mode,dim):
    # print(df)
    sns.set(style="whitegrid", palette="pastel", font_scale=1.2)
    map_mode = {'ZERO_SHOT': 'Zero-Shot', 'FINETUNE':'Finetune'}
    mode = map_mode[mode]
    model_name_mapping = {'mapzero': 'MapZero', 'smartmap': 'SmartMap'}
    arch_mapping = {'MESH': 'Mesh', 'OH_TOR_DIAG': 'OH+Tor+Diag', 'ONE_HOP': 'One-Hop'}

    df['model_name'] = df['model_name'].map(model_name_mapping)
    df['arch_interconnections'] = df['arch_interconnections'].map(arch_mapping)
    print(df)

    arch_order = ['OH+Tor+Diag', 'One-Hop', 'Mesh']

    df.rename(columns={'model_name':'Method'},inplace=True)

    plt.figure(figsize=(7, 4))
    sns.barplot(x='arch_interconnections', y='mapping_is_valid', hue='Method', data=df, hue_order=['MapZero', 'SmartMap'], order=arch_order)

    plt.xlabel('Architecture Interconnections')
    plt.ylabel('Valid Mapping Rate')
    plt.show()

def get_df_valid_mapping_rate_by_mapping_type(df:pd.DataFrame,col_type):
    df_vm = df[df['test_mode'] == col_type ][['model_name','arch_dims','arch_interconnections','mapping_is_valid']].groupby(['model_name','arch_dims','arch_interconnections']).mean()
    
    return df_vm.reset_index()

def get_df_used_pes_rate_by_mapping_type(df:pd.DataFrame):
    df_mt = df[df['mapping_is_valid'] == True]
    df_mt = df_mt[df_mt.duplicated(subset=['arch_dims','arch_interconnections','dfg_name','test_mode'], keep=False)]
    for arch_dims in df_mt['arch_dims'].unique():
        str_arch_dims = arch_dims.strip(')').strip('(')
        x , y = str_arch_dims.split(',')
        total_PES = int(x) * int(y)
        df_mt[df_mt['arch_dims'] == arch_dims]['used_PEs'] = df_mt[df_mt['arch_dims'] == arch_dims]['used_PEs']/total_PES
    
    print(df_mt[['used_PEs','model_name']].groupby('model_name').mean())


def get_df_mean_time(df:pd.DataFrame):
    df_mt = df[df['mapping_is_valid'] == True]
    df_mt = df_mt[df_mt.duplicated(subset=['arch_dims','arch_interconnections','dfg_name','test_mode'], keep=False)]
    print(df_mt[['mapping_time','model_name']].groupby('model_name').mean())
  
df = pd.read_csv('results/mapping_results.csv')
aux_df = df[df['mapping_is_valid'] == True]
print(aux_df.groupby('model_name').count()['mapping_is_valid'])


test_modes = [
    'ZERO_SHOT',
    'FINETUNE'
]

get_df_mean_time(df)
get_df_used_pes_rate_by_mapping_type(df)
for mode in test_modes:
    print('-'*100)
    print(mode)  
    print()
    temp_df = get_df_valid_mapping_rate_by_mapping_type(df,mode)
    print(temp_df)
    gen_graphics(temp_df[temp_df['arch_dims'] == "(4, 4)"],mode,'4')
    if mode =="ZERO_SHOT":
        gen_graphics(temp_df[temp_df['arch_dims'] == "(8, 8)"],mode,'8')
    print()
    print('-'*100)

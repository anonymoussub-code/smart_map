import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('results/train_results.csv')

# Create a boolean filter for the rows where model_name is 'yoto_mapzero'
filter_condition = df['model_name'] == 'yoto_mapzero'

# Update the 'model_name' column for those rows
df.loc[filter_condition, 'model_name'] = 'smartmap'

# Print the updated DataFrame
print(df['model_name'].unique())
df.to_csv('results/train_results.csv')

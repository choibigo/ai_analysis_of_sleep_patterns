import pandas as pd

# Read the data from the CSV file
df = pd.read_csv('origin_regression_testset.csv')

# Separate the "Age" column from the features
age_column = df['Age']
features = df.drop(columns=['Age'])

# Apply Min-Max normalization to each feature column
normalized_features = (features - features.min()) / (features.max() - features.min())

# Combine the "Age" column with the normalized features
normalized_df = pd.concat([age_column, normalized_features], axis=1)

# Save the normalized DataFrame to a new CSV file
normalized_df.to_csv('origin_regression_testset_normalized.csv', index=False)

# Display the normalized DataFrame
print(normalized_df)
"""
Just a simple helper file. Sometimes it is possible due to numeric instability or rounding issues that the obtained pareto-fronts of the exact methods contain duplicates
or solutions that are not entirely pareto-efficient. This script reads the outputted pareto-front (in the file file_path) and corrects it
"""

import pandas as pd
import os, config

# Read the CSV file into a pandas DataFrame
file_path = os.path.join(config.DIR_PARETO_FRONT_FULL, "/ParetoFront_Day11_BT1_4_BT2_4_BT4_2.csv")
df = pd.read_csv(file_path, sep=";")

# Function to check Pareto dominance
def is_pareto_efficient(costs, peak_load, df):
    """
    Check if a point is Pareto-efficient given Costs and Peak Load values.
    """
    return not any(
        (df['Costs'] < costs) & (df['Peak Load'] <= peak_load) |
        (df['Costs'] <= costs) & (df['Peak Load'] < peak_load)
    )

# Apply the Pareto-efficiency filter
pareto_efficient_mask = df.apply(lambda row: is_pareto_efficient(row['Costs'], row['Peak Load'], df), axis=1)
pareto_efficient_df = df[pareto_efficient_mask]

# Save the Pareto-efficient DataFrame to a new CSV file
output_file_path = file_path.replace(".csv", ".csv")
pareto_efficient_df.to_csv(output_file_path, sep=";", index=False)

# Display the resulting DataFrame
print("Pareto-Efficient DataFrame:")
print(pareto_efficient_df)




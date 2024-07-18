"""
This file contains the code to plot the results from wandb after downloading the runs as csv files.
Modify the individual dataframes in accordance with the column names and run the file.
"""


import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file into DataFrame
df = pd.read_csv(r'D:\Master-Thesis\msc2023_jayakumar\fasterRCNN\sgg\FW-EpiRew.csv')

# Define decay parameter for EMA smoothing
alpha = 0.01  # Adjust alpha as needed, smaller alpha means more smoothing

# Compute time-weighted EMA for each model's reward values
df['model1_reward_smooth'] = df['ALE/Freeway-v5__NoFrameStack_FW_Baseline__CNN___ppo_CNN_FW__1__1712830692 - evaluation/episodic_return'].ewm(alpha=alpha).mean()
df['model2_reward_smooth'] = df['ALE/Freeway-v5__FW_10Mclean_ppo_FW_v1__1__1714322387 - evaluation/episodic_return'].ewm(alpha=alpha).mean()
df['model3_reward_smooth'] = df['ALE/Freeway-v5__OBJ_SLOT_FW_10Mobj_slot_SPI__2__1714323031 - evaluation/episodic_return'].ewm(alpha=alpha).mean()

# Create the plot
plt.figure(figsize=(8, 5))

plt.plot(df['global_step'], df['model1_reward_smooth'], label='Baseline CNN', color='blue')
plt.plot(df['global_step'], df['model2_reward_smooth'], label='Scene graph GNN', color='green')
plt.plot(df['global_step'], df['model3_reward_smooth'], label='Object-centric', color='red')

plt.title('Freeway/Epiodic Reward', fontsize=16)
plt.xlabel('Steps', fontsize=14)
plt.ylabel('Episodic Reward', fontsize=14)
plt.legend(fontsize = 12)
plt.ylim(0, 60)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Save the plot as a PDF file
plt.savefig(r"D:\Master-Thesis\msc2023_jayakumar\fasterRCNN\sgg\plots\FW-eval-ER.pdf", bbox_inches='tight', dpi=300)

# Show the plot (optional)
plt.show()
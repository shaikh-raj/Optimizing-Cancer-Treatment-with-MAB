import pandas as pd
from environment import ClinicalTrialEnv, MultiArmedBanditEnv

# Load clinical trial data from a CSV file
# The data should contain columns for treatment type, treatment status, budget, and time
data = pd.read_csv("data/cancer.csv")

# Instantiate the ClinicalTrialEnv environment
# This environment simulates a clinical trial where different treatments are applied to patients
env = ClinicalTrialEnv(data)

# Render the initial state of the clinical trial environment
# This will show the number of trials and success rates for each treatment type
env.render()

# Define unique treatment types from the dataset and the maximum number of patients per treatment arm
treatments = data['Treatment Type'].unique()  # List of unique treatments in the dataset
max_patients_per_arm = 100  # Limit on the number of patients per treatment arm

# Instantiate the Multi-Armed Bandit environment
# This environment uses a multi-armed bandit approach to allocate patients to different treatment arms
bandit_env = MultiArmedBanditEnv(treatments, max_patients_per_arm)

# Simulate the clinical trial by iterating over the data
# For each patient in the dataset, we allocate them to a treatment arm using the multi-armed bandit strategy
# The success, cost, and time for each treatment are recorded and used to update the environment's state
for i in range(len(data)):
    treatment = bandit_env.allocate_patient()  # Allocate a patient to a treatment arm
    if treatment is not None:
        # Extract the success status, cost, and time from the dataset for the current patient
        success = data.iloc[i]['Treatment status(0=Failure,1=Success)']
        cost = data.iloc[i]['budget(in dollars)']
        time = data.iloc[i]['Time(In days)']
        
        # Update the bandit environment with the treatment outcome (success/failure), cost, and time
        bandit_env.update_rewards(treatment, success, cost, time)

# After all patients have been processed, display the success rates for each treatment
success_rates, total_trials = bandit_env.get_success_rates()

# Print the success rates and number of trials for each treatment arm
print("\nSuccess Rates:")
for treatment, success_rate in success_rates.items():
    print(f"Treatment {treatment}: Success Rate: {success_rate:.4f}")

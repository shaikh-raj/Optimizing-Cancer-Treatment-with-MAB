import random
import numpy as np

class ClinicalTrialEnv:
    """
    A class representing a clinical trial environment where different treatment arms are evaluated.
    The environment tracks the performance (success/failure) of each treatment type and updates 
    the state of the environment as trials are conducted.

    Attributes:
    -----------
    treatments : pandas.DataFrame
        A DataFrame containing treatment data, including treatment type and success/failure status.
    bandit_size : numpy.ndarray
        Unique treatment types being tested in the clinical trial.
    state : dict
        A dictionary that tracks the performance (success/failure) of each treatment type.

    Methods:
    --------
    step(row_index)
        Simulates a single step in the clinical trial by evaluating the treatment at a given row index.
    
    reset()
        Resets the environment to its initial state, clearing all treatment trial results.
    
    render()
        Displays the overall results of the clinical trial, including success rates and number of trials for each treatment type.
    """

    def __init__(self, treatments):
        """
        Initializes the ClinicalTrialEnv with a dataset of treatments.

        Parameters:
        -----------
        treatments : pandas.DataFrame
            A DataFrame containing information about treatments, including the treatment type and 
            the success/failure status (0 = Failure, 1 = Success).
        """
        self.treatments = treatments
        self.bandit_size = self.treatments['Treatment Type'].unique()  # Get unique treatment types
        self.state = {}
        self.reset()

    def step(self, row_index):
        """
        Simulates one trial (step) in the environment, where a treatment is applied to a patient.

        Parameters:
        -----------
        row_index : int
            The index of the row in the treatments DataFrame to evaluate.

        Returns:
        --------
        tuple:
            - state (dict): The updated state of the environment.
            - reward (int): Reward of 1 for success, -1 for failure.
            - done (bool): Always returns False (for multi-step trials).
            - None: Placeholder for future debugging information.
        """
        # Get the treatment status (0 = Failure, 1 = Success) and treatment type from the DataFrame
        treatment_status = self.treatments.iloc[row_index]['Treatment status(0=Failure,1=Success)']
        selected_treatment = self.treatments.iloc[row_index]['Treatment Type']
        
        # Reward based on success (1 for success, -1 for failure)
        reward = 1 if treatment_status == 1 else -1
        
        # Update the state by appending the treatment result (success or failure)
        self.state[selected_treatment].append(treatment_status)
        
        # No "done" condition for now, so always False
        done = False
        return self.state, reward, done, None

    def reset(self):
        """
        Resets the environment to its initial state, clearing all the recorded trial results.

        Returns:
        --------
        dict:
            The initial state of the environment with empty lists for each treatment type.
        """
        # Initialize the state as an empty list for each treatment type
        self.state = {treatment_type: [] for treatment_type in self.bandit_size}
        return self.state

    def render(self):
        """
        Displays the results of the clinical trial, including the success rate and total number of trials 
        for each treatment type.
        """
        # Calculate total trials and total successes (returns) for each treatment
        total_trials = {treatment: len(self.state[treatment]) for treatment in self.bandit_size}
        total_returns = {treatment: sum(self.state[treatment]) for treatment in self.bandit_size}

        # Display trial results
        print("\n=== Clinical Trial Results ===")
        for treatment in self.bandit_size:
            # Calculate success rate for each treatment type
            success_rate = total_returns[treatment] / total_trials[treatment] if total_trials[treatment] > 0 else 0
            print(f"Treatment: {treatment}: Success Rate: {success_rate:.4f}, Trials: {total_trials[treatment]}")

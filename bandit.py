import random

class MultiArmedBanditEnv:
    """
    A class representing the Multi-Armed Bandit environment for allocating patients to different treatment arms
    based on an exploration-exploitation strategy. It tracks the number of patients assigned, successes, failures,
    time to discovery, and costs associated with each treatment.

    Attributes:
    -----------
    treatments : list
        A list of available treatment types.
    max_patients_per_arm : int
        Maximum number of patients that can be assigned to each treatment arm.
    patients_assigned : dict
        A dictionary that keeps track of the number of patients assigned to each treatment arm.
    successes : dict
        A dictionary that stores the number of successful treatments for each treatment arm.
    failures : dict
        A dictionary that stores the number of failed treatments for each treatment arm.
    total_patients : int
        Total number of patients in the trial.
    time_to_discovery : dict
        Tracks the time to the first successful treatment for each treatment arm.
    costs : dict
        Accumulates the cost of treatments for each arm.

    Methods:
    --------
    allocate_patient()
        Allocates a patient to a treatment arm based on the exploration-exploitation strategy.
    
    update_rewards(treatment, success, cost, time)
        Updates the environment's state based on the treatment outcome (success/failure), cost, and time.
    """
    
    def __init__(self, treatments, max_patients_per_arm):
        """
        Initializes the MultiArmedBanditEnv with a set of treatments and a maximum number of patients allowed per arm.

        Parameters:
        -----------
        treatments : list
            A list of treatment types available for the clinical trial.
        max_patients_per_arm : int
            Maximum number of patients that can be assigned to each treatment arm.
        """
        self.treatments = treatments
        self.max_patients_per_arm = max_patients_per_arm
        self.patients_assigned = {treatment: 0 for treatment in treatments}  # Tracks patients assigned to each arm
        self.successes = {treatment: 0 for treatment in treatments}  # Tracks successful treatments for each arm
        self.failures = {treatment: 0 for treatment in treatments}  # Tracks failed treatments for each arm
        self.total_patients = 0  # Total number of patients in the clinical trial
        self.time_to_discovery = {treatment: 0 for treatment in treatments}  # Time to first success for each arm
        self.costs = {treatment: 0 for treatment in treatments}  # Total costs for each arm

    def allocate_patient(self):
        """
        Allocates a patient to a treatment arm based on an epsilon-greedy exploration-exploitation strategy.

        The strategy balances between exploring random treatments (exploration) and exploiting the treatment 
        with the highest observed success rate (exploitation).

        Returns:
        --------
        str or None:
            The selected treatment arm, or None if the maximum number of patients per arm is reached.
        """
        epsilon = 0.1  # Exploration rate: 10% chance to randomly explore a treatment
        if random.random() < epsilon:
            # Randomly select a treatment (exploration)
            treatment = random.choice(self.treatments)
        else:
            # Exploit: select the treatment with the highest success rate
            success_rates = {treatment: self.successes[treatment] / (self.successes[treatment] + self.failures[treatment] + 1)
                             for treatment in self.treatments}
            treatment = max(success_rates, key=success_rates.get)

        # Ensure the treatment arm has not reached its patient capacity
        if self.patients_assigned[treatment] >= self.max_patients_per_arm:
            return None  # Return None if the treatment arm is full

        return treatment

    def update_rewards(self, treatment, success, cost, time):
        """
        Updates the internal state of the environment after a patient is treated, based on the success/failure 
        of the treatment, the cost, and the time taken.

        Parameters:
        -----------
        treatment : str
            The treatment arm to which the patient was assigned.
        success : bool
            A boolean indicating if the treatment was successful (True) or not (False).
        cost : int
            The cost incurred for the treatment.
        time : int
            The time taken for the treatment.

        Returns:
        --------
        None
        """
        if success:
            # Increment successes for the treatment
            self.successes[treatment] += 1
            # If it's the first success, record the time to discovery
            if self.successes[treatment] == 1:
                self.time_to_discovery[treatment] = time
        else:
            # Increment failures for the treatment
            self.failures[treatment] += 1

        # Update patient count and cost for the treatment
        self.patients_assigned[treatment] += 1
        self.costs[treatment] += cost
        self.total_patients += 1  # Increase total number of patients treated

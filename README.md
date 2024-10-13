# Optimizing Cancer Treatment with Multi-Armed Bandits

This project optimizes cancer treatment allocation using a Multi-Armed Bandit (MAB) approach. 

## Requirements
- Python 3.x
- Dependencies listed in `requirements.txt`

## How to Run
1. Install the required dependencies.
   ```bash
   pip install -r requirements.txt
   ```
2. Run the simulation.
   ```bash
   python main.py
   ```

## Project Structure
- `main.py`: Entry point to run the simulation.
- `environment.py`: Contains the `ClinicalTrialEnv` and `MultiArmedBanditEnv` classes.
- `bandit.py`: Implements the bandit algorithms for decision-making.
- `data/cancer.csv`: Dataset used for clinical trials.
- `utils/`: Utility functions (if any).

## Results
- Success rates, time-to-discovery, and cost-effectiveness for each treatment arm will be printed in the console.

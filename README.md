
# Health Study Analysis Project (Part 1: G & VG)

This repository holds the code and final results for the required health study data analysis assignment. The project focuses on performing descriptive statistics, simulations, and advanced statistical inference on health data (health_study_dataset.csv).

The code is structured modularly using a Python class (HealthDataAnalyzer) to keep the analysis logic separate from the presentation layer (the Jupyter Notebook), following common industry standards.

# Project Structure

health_study_del_1
├── src/
│   ├── __init__.py               
│   └── health_analyzer.py        # Python Class with all cleaning and analysis logic
├── data/                         # Holds the raw CSV data
│   └── health_study_dataset.csv
├── plots/                        # Generated output (PNG graphs)
├── Health_Study_Notebook.ipynb   # Main document showing results and calling the class
├── README.md                     # This file
└── requirements.txt              # Project dependencies

# Setup and Execution
The project environment was built using Python 3.11.9, as required for our course setup.

Install Dependencies: All necessary libraries (Pandas, NumPy, SciPy, etc.) are listed in requirements.txt.

pip install -r requirements.txt
(Note: The code handles the creation of the plots/ folder automatically upon execution.)

Run the Analysis: Open the Health_Study_Notebook.ipynb file in your Jupyter environment and run the cells sequentially. The notebook imports the HealthDataAnalyzer class, executes the cleaning steps, runs the statistics, and displays all required graphs and conclusions.

# Analysis Overview

The analysis covers all required levels, G (Godkänd) and VG (Väl Godkänd).

G (Basic Analysis)
***Descriptive Statistics:*** Calculation of mean, median, min, and max for key health metrics (age, weight, systolic_bp, etc.).

***Visualization:*** Creation of 3 distinct plots (histogram, boxplot, and bar chart) saved to the plots/ directory and displayed in the Notebook.

***Simulation:*** Comparison of the actual disease prevalence against a Monte Carlo simulation (N=1000).

***Confidence Interval (CI):*** Calculation of a 95% CI for mean systolic blood pressure using Normal Approximation.

***Hypothesis Testing:*** Execution of a one-sided Welch's T-test to determine if smokers have higher mean blood pressure than non-smokers.

VG (Advanced Analysis)
***CI Comparison:*** The 95% CI is calculated using a second, more robust method (Bootstrap) and compared against the Normal Approximation results.

***Statistical Power:*** A simulation is performed to estimate the statistical Power of the hypothesis test, assessing the test's reliability to detect the observed difference.

***Method Justification:*** Comprehensive justification for all methodological choices (e.g., why Welch's T-test was used, the value of Bootstrap) is included in a final Markdown section.
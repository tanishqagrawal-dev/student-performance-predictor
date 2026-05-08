import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate 500 samples
n_samples = 500

# Features
study_hours = np.random.uniform(1, 10, n_samples)
attendance = np.random.uniform(60, 100, n_samples)
assignment_score = np.random.uniform(40, 100, n_samples)
previous_marks = np.random.uniform(30, 100, n_samples)

# Target: Final Marks (Linear combination + noise)
# Weights: Hours(4), Attendance(0.2), Assignment(0.1), Previous(0.3) - scaled roughly to 100
final_marks = (study_hours * 4.5) + (attendance * 0.15) + (assignment_score * 0.1) + (previous_marks * 0.2) + np.random.normal(0, 2, n_samples)

# Clip marks to 0-100 range
final_marks = np.clip(final_marks, 0, 100)

# Target: Status (Pass/Fail) - Pass if marks >= 40
status = (final_marks >= 40).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Study_Hours': np.round(study_hours, 1),
    'Attendance': np.round(attendance, 1),
    'Assignment_Score': np.round(assignment_score, 1),
    'Previous_Marks': np.round(previous_marks, 1),
    'Final_Marks': np.round(final_marks, 1),
    'Status': status
})

# Save to CSV
df.to_csv('sample_dataset.csv', index=False)
print("Dataset created successfully!")

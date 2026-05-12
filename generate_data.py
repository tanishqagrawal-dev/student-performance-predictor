import pandas as pd
import numpy as np

# Set seed so that the data stays the same every time we run it
np.random.seed(42)

# We want 1000 students in our dataset for better training
n_samples = 1000

# Generating random features for students
# Study hours range from 0 to 12 hours a day
study_hours = np.random.uniform(0, 12, n_samples)
# Attendance is between 30% and 100%
attendance = np.random.uniform(30, 100, n_samples)
# Scores for assignments and previous marks (20 to 100)
assignment_score = np.random.uniform(20, 100, n_samples)
previous_marks = np.random.uniform(20, 100, n_samples)

# Function to calculate the base marks based on some weights
def calculate_base_score(sh, at, sc, pm):
    # These weights are decided based on how important each factor is
    score = (sh * 2.9) + (at * 0.20) + (sc * 0.25) + (pm * 0.30)
    
    # Adding some bonus marks for high performers
    bonus = 0
    if at >= 90: bonus += 3
    if sc >= 90: bonus += 3
    if pm >= 85: bonus += 4
    
    return np.clip(score + bonus, 0, 100)

# Generating final marks with some random noise to make it realistic
final_marks = []
for i in range(n_samples):
    base = calculate_base_score(study_hours[i], attendance[i], assignment_score[i], previous_marks[i])
    # Adding a bit of randomness (standard deviation of 2.5)
    final_marks.append(np.clip(base + np.random.normal(0, 2.5), 0, 100))

final_marks = np.array(final_marks)

# Putting everything together into a Pandas DataFrame
df = pd.DataFrame({
    'Study_Hours': np.round(study_hours, 1),
    'Attendance': np.round(attendance, 1),
    'Assignment_Score': np.round(assignment_score, 1),
    'Previous_Marks': np.round(previous_marks, 1),
    'Final_Marks': np.round(final_marks, 1),
    'Status': (final_marks >= 40).astype(int) # 1 if Pass (>=40), else 0
})

# Save the dataset to a CSV file
df.to_csv('sample_dataset.csv', index=False)
print(f"Done! Generated {n_samples} student records in 'sample_dataset.csv'")

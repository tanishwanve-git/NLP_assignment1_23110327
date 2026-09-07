import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime, timedelta

def create_gantt_chart(output_path="gantt_chart.png"):
    start_date = datetime(2026, 8, 30)
    
    tasks = [
        {"Task": "Project Proposal & Planning", "Start": 0, "Duration": 1},
        {"Task": "Literature Review & Report Intro", "Start": 1, "Duration": 1},
        {"Task": "Data Acquisition & EDA", "Start": 2, "Duration": 1},
        {"Task": "Classical Baselines (TF-IDF + LR)", "Start": 3, "Duration": 2},
        {"Task": "LLM Fine-Tuning (DistilBERT)", "Start": 4, "Duration": 3},
        {"Task": "Evaluation & Results Compilation", "Start": 6, "Duration": 1},
        {"Task": "Gradio Web App Development", "Start": 7, "Duration": 2},
        {"Task": "Final Report & Viva Prep", "Start": 8, "Duration": 1}
    ]

    # Convert to DataFrame
    df = pd.DataFrame(tasks)
    
    # Calculate absolute dates
    df['Start_Date'] = df['Start'].apply(lambda x: start_date + timedelta(days=x))
    df['End_Date'] = df.apply(lambda row: row['Start_Date'] + timedelta(days=row['Duration']), axis=1)

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    for i, task in enumerate(df.itertuples()):
        ax.barh(task.Task, task.Duration, left=task.Start_Date, color='skyblue', edgecolor='black')

    # Formatting
    ax.set_xlabel('Timeline')
    ax.set_title('NLP Assignment: Toxic Comment Classification Project Schedule')
    
    # Format x-axis to show dates nicely
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    
    # Invert y-axis so the first task is at the top
    ax.invert_yaxis()
    
    # Add gridlines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Gantt chart successfully saved to {output_path}")

if __name__ == "__main__":
    create_gantt_chart()

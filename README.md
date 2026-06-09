[README.md](https://github.com/user-attachments/files/28743451/README.md)
# Deniz_Berke_Ozsoy_AI_Agent_Test_V2
This project is done accordingly to the requirements of AI_Agent_Test_V2

# AI-Powered Data Analysis Assistant

## Persistent Google Drive Location
## The folder should be uploaded to Drive and run with the Google Colab without changing name and file structure 
## You should inject your secret keys to Google Colab as a secret with using key icon (due to privacy considerations my personal API keys did not provided)

All generated files and related outputs are stored under:

```text
/content/drive/MyDrive/multi_agent_data_analysis_assistant_colab_package
```

Recommended folder structure:

```text
multi_agent_data_analysis_assistant_colab_package/
├── multi_agent_data_analysis_assistant_colab_enhanced.ipynb
├── utils.py
├── tools.py
├── agent.py
├── evaluate.py
├── README.md
├── user_profile.json
├── data/
│   ├── vehicles.xlsx
│   ├── holidays.xlsx
│   └── weather.xlsx
└── outputs/
    ├── charts/
    │   ├── vehicle_consumption_comparison_all.png
    │   └── vehicle_consumption_comparison_all.pdf
    └── reports/
        ├── evaluation_summary.csv
        ├── evaluation_details.csv
        ├── evaluation_report.html
        └── evaluation_metrics.png
```

## Overview

This project implements a hierarchical multi-agent data analysis assistant for Excel-based natural-language queries.

The system can answer Turkish questions about:

- Vehicle fuel consumption
- Official holidays
- Istanbul historical weather averages
- Future weather forecasts using an external fallback mock tool
- Vehicle fuel-consumption visualization

## Architecture

The system uses a hierarchical multi-agent architecture:

1. `GuardrailValidator`
2. `PlannerAgent`
3. `ExecutorAgent`
4. `EditorCriticAgent`

```text
User Query
   ↓
GuardrailValidator
   ↓
PlannerAgent
   ↓
ExecutorAgent
   ↓
EditorCriticAgent
   ↓
Final Turkish Answer
```

## Key Enhancements

- All modules are written directly to the requested Drive project folder.
- All charts are exported to `outputs/charts`.
- All evaluation artifacts are exported to `outputs/reports`.
- Stored vehicle preferences now include fallback behavior. If the user prefers SUVs but no SUV exists in the dataset, the agent retries with all vehicles and explains this transparently.
- The evaluation suite now exports CSV, HTML, and PNG benchmark reports.
- The UI renders inline charts and expandable multi-agent traces.

## Required Excel Files

The project supports the following naming convention:

- `holidays.xlsx`
- `vehicles.xlsx`
- `weather.xlsx`

## How to Run

1. Open the notebook in Google Colab.
2. Upload the Excel files or place them in the Drive project folder.
3. Run the setup cell.
4. Run all `%%writefile` cells.
5. Run the smoke test.
6. Run the evaluation suite.
7. Launch the interactive UI.

## Evaluation Metrics

The automated suite reports:

- Intent Accuracy
- Tool Selection Accuracy
- Action Completion Rate
- Guardrail Trigger Rate
- Fallback Robustness

## Security

The system blocks prompt-injection and destructive requests such as:

- Ignore previous instructions
- Reveal hidden prompt
- Delete system files
- Leak API keys
- Read `.env`

## Visualization

Vehicle fuel-consumption charts are generated using:

- seaborn
- matplotlib
- 300 DPI export
- PNG and PDF output
- whitegrid academic theme

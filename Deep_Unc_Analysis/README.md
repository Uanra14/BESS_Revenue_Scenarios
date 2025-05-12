# Deep Uncertainty Analysis

This project performs parameter sensitivity analysis on an Excel-based financial model. It systematically varies input parameters, runs the model for each combination, and generates visualizations of the results.

## Project Structure

```
deep_uncertainty_analysis/
├── models/                    # Excel model files and configurations
│   ├── input/                # Input Excel models
│   └── output/               # Output Excel files from runs
├── analysis/                  # Analysis code and notebooks
│   ├── notebooks/            # Jupyter notebooks
│   └── src/                  # Source code
├── config/                   # Configuration files
├── results/                  # Analysis results
│   ├── raw/                 # Raw output data
│   └── plots/               # Generated plots
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## Setup

1. Create a new Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your Excel model in `models/input/`

4. Configure parameter ranges in `config/parameter_ranges.json`

## Usage

1. Run parameter analysis:
   - Open `analysis/notebooks/parameter_analysis.ipynb`
   - Configure your parameter ranges and scenarios
   - Run the analysis

2. Generate visualizations:
   - Open `analysis/notebooks/visualization.ipynb`
   - Select the results to visualize
   - Generate plots

## Dependencies

- numpy: For numerical computations
- pandas: For data manipulation
- matplotlib: For plotting
- ema_workbench: For exploratory modeling and analysis
- seaborn: For statistical visualizations
- plotly: For interactive plots
- openpyxl: For Excel file handling

## Workflow

1. **Parameter Configuration**
   - Define parameter ranges in `config/parameter_ranges.json`
   - Configure scenarios in `config/scenarios.json`

2. **Model Execution**
   - Run the Excel model with different parameter combinations
   - Collect results from each run

3. **Analysis**
   - Process the collected results
   - Generate visualizations
   - Analyze parameter sensitivity

4. **Output**
   - Save raw results in `results/raw/`
   - Save generated plots in `results/plots/`

## Development Guidelines

1. **Code Organization**
   - Keep model-specific code in the `src/models` directory
   - Place utility functions in `src/utils`
   - Store visualization code in `src/visualization`

2. **Notebooks**
   - Use notebooks in `notebooks/analysis` for final analysis
   - Use notebooks in `notebooks/exploration` for initial data exploration
   - Keep notebooks clean and well-documented

3. **Data Management**
   - Store raw data in `data/raw`
   - Save processed data in `data/processed`
   - Use relative paths in code

4. **Version Control**
   - Keep large data files out of version control
   - Use `.gitignore` for sensitive information
   - Document data sources and processing steps

## Contributing

1. Create a new branch for each feature or bug fix
2. Follow the project structure guidelines
3. Update documentation as needed
4. Test changes before submitting pull requests 
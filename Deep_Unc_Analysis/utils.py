### --- ### 
# Use this file to store helper functions in this file to improve readibility of the main code
### --- ### 

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import pandas as pd
import plotly.express as px
from matplotlib.animation import FuncAnimation, PillowWriter

import ema_workbench.analysis.feature_scoring as feature_scoring
from ema_workbench import ema_logging, perform_experiments, save_results, SequentialEvaluator, load_results, Samplers
from ema_workbench.analysis import lines, Density
from ema_workbench.connectors.excel import ExcelModel
from ema_workbench.analysis.feature_scoring import RuleInductionType
from ema_workbench.em_framework.model import AbstractModel

### --- ### GLOBAL VARIABLES ### --- ###

model_directory = f"{os.getcwd()}/model"

### --- ### CREATE AND RUN MODEL FUNCTIONS ### --- ### ------------------------------------------------------------------------------------------------------------------------------------------------------------------

def instantiate_model(model_file):
    '''
    Instantiate an ExcelModel object, and set the default sheet to "General Inputs".
    @param model_file: The name of the excel file to instantiate the model from.
    '''
    ema_logging.log_to_stderr(level = ema_logging.INFO)

    model = ExcelModel("MultiAsset", wd = model_directory, model_file = model_file)
    model.default_sheet = "General Inputs"

    return model

def reshape_ts_outcomes(outcomes):
    """
    Reshape the time series outcomes to a format that can be used for plotting.
    @param outcomes: The outcomes to reshape.
    """
    for name, value in outcomes.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            # Assuming the first dimension is the number of scenarios
            num_scenarios = value.shape[0]
            num_timesteps = np.prod(value.shape[1:])

            outcomes[name] = value.reshape((num_scenarios, num_timesteps))
    return outcomes

def run_experiments(model, num_experiments, num_policies = None):
    # Ensure the destination folder exists
    output_folder = "model/outputs"
    os.makedirs(output_folder, exist_ok=True)

    evaluator = SequentialEvaluator(model)
    evaluator.initialize()
    results = perform_experiments(model, scenarios = num_experiments, policies = num_policies, reporting_interval = None, evaluator = evaluator, uncertainty_sampling = Samplers.LHS)
    evaluator.finalize()

    x, o = results
    x['scenario'] = range(len(x))
    o = reshape_ts_outcomes(o)
    save_results(results, f"model/outputs/experiments.tar.gz")
    
    # clean up the model
    return x, o

def load_results_from_file(file_path):
    return load_results(file_path)

### --- ### PLOTTING FUNCTIONS ### --- ### ------------------------------------------------------------------------------------------------------------------------------------------------------------------

def create_line_graph(x, o, case_list, outcome_name):
    """
    Create a line graph of the specified outcome over time.

    Parameters:
    - x (pd.DataFrame): The input features.
    - o (dict): The outcomes dictionary from the model run.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object containing the plot.
    """
    # Check if there is more than one case
    if len(case_list) > 1:
        fig, ax = lines(x, o, legend=True, outcomes_to_show=outcome_name, density = Density.KDE)
    else:
        fig, ax = lines(x, o, legend=True, outcomes_to_show=outcome_name, density = Density.KDE)


    fig.set_size_inches(14, 7)
    ax = fig.get_axes()
    sns.set_style('whitegrid')

    # Update line properties
    for axis in ax:
        for line in axis.get_lines():
            line.set_alpha(0.65)
            line.set_linewidth(1) 
        axis.margins(y=0)

    # Clean up spines and grid lines
    for axis in ax:
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['left'].set_linewidth(0.5)
        axis.spines['bottom'].set_linewidth(0.5)
        axis.xaxis.set_tick_params(width=0.5)
        axis.yaxis.set_tick_params(width=0.5)
        axis.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)  # Light dashed grid lines

    # Set titles and labels with larger font size
        ax[0].set_title(outcome_name, fontsize=16, pad=20)
        ax[1].set_title('Density', fontsize=16, pad=20)
        ax[0].set_xlabel('Time (months)', fontsize=14, labelpad=10)
        ax[0].set_xlabel('Time (months)', fontsize=14, labelpad=10)
        ax[0].set_ylabel(outcome_name, fontsize=14, labelpad=10)

    # Add and style legend
    for axis in ax:
        legend = axis.get_legend()
        if legend:
            legend.set_frame_on(False)  # Remove legend frame
            legend.get_frame().set_alpha(0.2)  # Slight transparency

    sns.despine()
    plt.close(fig)

    return fig

def create_animation(o):
    """
    Animation function for updating the plot with each frame.
    @param frame: The current frame number.
    @param lines: The lines to update in the plot.
    @param outcome_data: The outcome data to plot.
    @param time_data: The time data to plot.
    """

    def init_animation():
        """
        Initialize the animation by clearing the lines.
        """
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(frame):
        """
        Animation function for updating the plot with each frame.
        @param frame: The current frame number.
        """
        print(f"Animating frame {frame+1}/{len(lines)}")

        for i in range(frame + 1):
            lines[i].set_data(time_data, outcome_data[i])
        return lines

    outcome_name = "Rolling_Equity_IRR"
    time_data = o["TIME"][0]  # Use the first array of time (assuming the same for all model runs)
    outcome_data = o[outcome_name]

    # Limit to the first 10 lines for debugging
    num_lines_to_plot = 200
    outcome_data = outcome_data[:num_lines_to_plot]  # Select only the first 10 model runs

    with plt.ioff():
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Labeling the axes and title
        ax.set_xlabel('Time')
        ax.set_ylabel(outcome_name)
        ax.set_title(f"Time-Tracked Outcome: {outcome_name}")

        # Line plot (initialize empty lines for each model run)
        lines = [ax.plot([], [])[0] for i in range(outcome_data.shape[0])]

        # Initialize the axes limits
        ax.set_xlim(time_data[0], time_data[-1])
        ax.set_ylim(np.min(outcome_data), np.max(outcome_data))
    
    anim = FuncAnimation(fig, animate, frames=len(lines), init_func = init_animation, blit=True, interval=5)

    writer = PillowWriter(fps=25)  # Set the frames per second
    anim.save('time_tracked_outcome_animation_1000_lines.gif', writer=writer)

    print("Animation saved as 'time_tracked_outcome_animation_1000_lines.gif'")

def plot_npv_kde_by_case(o, num_experiments, num_policies, cases):
    """
    Plot the NPV of the equity for each scenario using a density plot.

    Parameters:
    - o (dict): The outcomes dictionary from the model run.
    - num_policies (int): The number of policies in the model.
    - num_experiments (int): The number of experiments in the model.
    - cases (list): List of the names of the cases.

    Returns:
    - fig (plotly.graph_objects.Figure): The figure object containing the density plot.
    """
    if len(o['Equity_NPV']) != len(cases) * num_experiments:
        raise ValueError("Mismatch between the number of NPV values and the expected length based on case_list and num_experiments.")

    # Extract NPV values from the dictionary `o`
    npv_values = np.array(o['Equity_NPV']).astype(float)

    scenarios = []
    for i in range(num_policies):
        start_index = i * num_experiments
        end_index = start_index + num_experiments
        scenarios.append(npv_values[start_index:end_index])

    with plt.ioff():
        fig = ff.create_distplot(
            scenarios, 
            group_labels = cases,
            #colors=["#1c33ff","#ff1930", "#ffb01c"],
            show_hist=False,
            show_rug=False,
        )

        fig.update_layout(
            title='Density Plot of Equity NPV per Scenario',
            xaxis_title='NPV [kEUR]',
            yaxis_title='Density',
            legend_title='Scenario',
            width=800,
            height=600,
        )

    return fig

def plot_npv_boxplot(o, case_series, x_axis_order = None):
    # Extract NPV values
    npv_values = np.array(o['Equity_IRR']).astype(float)
    
    # make the case_series an array of the same length as npv_values by repeating the first value
    case_series = np.array(case_series).astype(str)
    case_series = np.repeat(case_series, len(npv_values) // len(case_series))

    # Create a DataFrame
    data = pd.DataFrame({
        'IRR [Decimal]': npv_values,
        'Scenario': case_series
    })
    print(case_series)
    # Apply explicit ordering if provided
    if x_axis_order != None:
        data['Scenario'] = pd.Categorical(data['Scenario'], categories=x_axis_order, ordered=True)
    else:
        x_axis_order = data['Scenario'].unique()
    
    fig = px.box(
        data,
        x='Scenario',
        y='IRR [Decimal]',
        color='Scenario',
        title='Box Plot of Equity IRR per Scenario',
        color_discrete_sequence=[
            "#1c33ff", "#ff1930", "#ffb01c", "#7519ff", "#ff6f1c", 
            "#a3ff19", "#19ffe8", "#e819ff", "#a5a7dd", "#565a8c", 
            "#930700", "#2ade72"
        ]
    )

    # Update layout with size parameters
    fig.update_layout(
        width=800,
        height=600,
        xaxis_title="Scenario",
        yaxis_title="IRR [Decimal]"
    )

    return fig

def rank_features_IRR(x, o):
    """	
    Rank features based on their importance in predicting the IRR.

    Parameters:
    - x (pd.DataFrame): The input features.
    - o (dict): The outcomes dictionary from the model run.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object containing the plot.
    """
    # Define the target variable
    y = o["Equity_IRR"] > 0.15

    # Get feature scores
    fs, alg = feature_scoring.get_ex_feature_scores(x, y, mode=RuleInductionType.CLASSIFICATION)

    # Rename columns
    fs.columns = ['Score']
    fs.index.name = 'Feature'

    # Filter out unwanted rows
    fs_filtered = fs[(fs.index != 'policy') & (fs.index != 'Case')]

    with plt.ioff():
        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x = fs_filtered.index, y = fs_filtered['Score'], ax=ax, palette="viridis_r", hue=fs_filtered.index)
        ax.set_title('Feature Importance Scores', fontsize=16)
        ax.set_xlabel('Feature', fontsize=14)
        ax.set_ylabel('Score', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(False)
        sns.despine()
    
    plt.close(fig)  # Close the plotly figure to avoid displaying it in Jupyter Notebook

    return fig
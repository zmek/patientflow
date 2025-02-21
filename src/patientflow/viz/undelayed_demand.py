import pandas as pd
import numpy as np
from patientflow.viz.arrival_rates import plot_arrival_rates


def generate_plot(df: pd.DataFrame,
                    x1: float,
                    y1: float,
                    x2: float,
                    y2: float,                  
                  time_interval: int = 60,
                  start_plot_index: int = 0, 
                  num_days: int = None
                  ):

    if not num_days:
        num_days = len(np.unique(df.index.date))

    # Set the plot to start at the 8th hour of the day (if not set the function will default to starting at midnight
    start_plot_index = 8

    # plot for weekdays
    title = f'Hourly arrival rates of admitted patients starting at {start_plot_index} am from {df.index.date.min()} to {df.index.date.max()}'
    fig = plot_arrival_rates(df, 
                    title, 
                    time_interval=60, 
                    start_plot_index=start_plot_index, 
                    num_days=num_days,
                    return_figure=True)
    
    return fig


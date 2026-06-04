#!/usr/bin/python
# %%
import sys

import numpy as np
import ogstools as ot
import pandas as pd
from matplotlib import pyplot as plt
from ogstools.logparser import (
    analysis_convergence_newton_iteration,
    analysis_time_step,
    fill_ogs_context,
    parse_file,
    time_step_vs_iterations,
)

# %%
# Read logfiles from command line arguments
if len(sys.argv) < 2:
    print("Usage: python log_compare.py <logfile1> [logfile2] ...")
    sys.exit(1)

logfiles = sys.argv[1:]
error_metric = "dx_x"
# %%
records = []
for logfile in logfiles:
    records.append(parse_file(logfile))
df_records = []
for record in records:
    df_records.append(pd.DataFrame(record))
df_logs = []
for df_record in df_records:
    df_logs.append(fill_ogs_context(df_record))
# %%
df_it = []
df_ts = []
df_newton = []
error_metrics = ["dx_x"]  # Default error metric
for df_log in df_logs:
    df_it.append(time_step_vs_iterations(df_log))
    df_ts.append(analysis_time_step(df_log))
    df_ts[-1] = df_ts[-1].loc[0]
    df_newton.append(analysis_convergence_newton_iteration(df_log))
    # Check if residual_norm is available in the parsed columns
    if "residual_norm" in df_log.columns:
        # Find which components have actual data for residual_norm
        residual_components_with_data = []
        if "component" in df_newton[-1]["residual_norm"].index.names:
            all_comps = (
                df_newton[-1]["residual_norm"]
                .index.get_level_values("component")
                .unique()
                .tolist()
            )
            for c in all_comps:
                try:
                    data = (
                        df_newton[-1]["residual_norm"].unstack("component")[c].dropna()
                    )
                    if len(data) > 0:
                        residual_components_with_data.append(c)
                except:
                    pass
        if "residual_norm" not in error_metrics:
            error_metrics.append("residual_norm")
# %%
# %%
# Create a scrollable window with all plots
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


def create_scrollable_plot_window():
    """Create a Tkinter window with scrollable matplotlib plots."""
    root = tk.Tk()
    root.title("Log Comparison - All Plots")
    root.geometry("1200x800")

    # Create main canvas with scrollbar (without toolbar in scrollable area)
    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create matplotlib figure for all plots with constrained layout
    fig = Figure(
        figsize=(14, 7 * 10),
        constrained_layout=True,
    )  # 10 plots max, scrollable
    fig.suptitle("Log Comparison - All Metrics", fontsize=16, fontweight="bold")

    # Plot 1: Iteration number vs time step
    ax1 = fig.add_subplot(10, 1, 1)
    for i, entry in enumerate(logfiles):
        ax1.plot(df_it[i]["iteration_number"], label=entry)
    ax1.legend()
    ax1.set_xlabel("time step")
    ax1.set_ylabel("iteration number")
    ax1.set_title("Iteration Number vs Time Step")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative iteration number
    ax2 = fig.add_subplot(10, 1, 2)
    for i, entry in enumerate(logfiles):
        ax2.plot(df_it[i]["iteration_number"].cumsum(), label=entry)
    ax2.legend()
    ax2.set_xlabel("time step")
    ax2.set_ylabel("iteration number")
    ax2.set_title("Cumulative Iteration Number")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Step start time
    ax3 = fig.add_subplot(10, 1, 3)
    for i, entry in enumerate(logfiles):
        ax3.plot(df_it[i]["step_start_time"], label=entry)
    ax3.legend()
    ax3.set_xlabel("time step")
    ax3.set_ylabel("time / s")
    ax3.set_title("Step Start Time")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Step size
    ax4 = fig.add_subplot(10, 1, 4)
    for i, entry in enumerate(logfiles):
        ax4.plot(df_ts[i]["step_size"], label=entry)
    ax4.legend()
    ax4.set_xlabel("time step")
    ax4.set_ylabel("time step size / s")
    ax4.set_title("Step Size vs Time Step")
    ax4.grid(True, alpha=0.3)

    # Plot Newton iteration metrics (dx_x and optionally residual_norm)
    if len(df_newton) > 0:
        subplot_idx = 5  # Start at subplot 5

        # Determine number of components (same for all metrics)
        first_metric = error_metrics[0]
        dx_x_index = df_newton[0][first_metric].index
        if "component" in dx_x_index.names:
            components = dx_x_index.get_level_values("component").unique()
            # Filter out component -1 if it exists (often represents total/aggregate)
            components = [c for c in components if c != -1]
        else:
            components = ["default"]
        n_components = len(components)
        n_metrics = len(error_metrics)

        # Total subplots available (from 5 onwards, we have plenty with 12 total)
        total_available = 12 - subplot_idx + 1  # = 8 subplots available

        # Strategy: Show all dx_x components separately, then residual_norm if space
        plots_to_make = []

        if "dx_x" in error_metrics:
            # Add all dx_x components
            for comp in components:
                if len(plots_to_make) < total_available:
                    plots_to_make.append(("dx_x", comp))

        if "residual_norm" in error_metrics and residual_components_with_data:
            # Add residual_norm components with data (usually just component -1)
            for comp in residual_components_with_data:
                if len(plots_to_make) < total_available:
                    plots_to_make.append(("residual_norm", comp))

        # Find all unique time_step/iteration combinations
        all_time_steps = []
        all_iteration_numbers = []
        for i in range(len(logfiles)):
            for comp in components:
                metric = error_metrics[0]
                if (
                    comp == "default"
                    or "component" not in df_newton[i][metric].index.names
                ):
                    dx_x = df_newton[i][metric].dropna()
                else:
                    dx_x = df_newton[i][metric].unstack("component")[comp].dropna()
                all_time_steps.extend(dx_x.index.get_level_values("time_step").tolist())
                all_iteration_numbers.extend(
                    dx_x.index.get_level_values("iteration_number").tolist()
                )

        unique_combinations = sorted(set(zip(all_time_steps, all_iteration_numbers)))

        # Create plots for each metric-component combination
        for metric, comp in plots_to_make:
            if subplot_idx > 10:  # 10-row grid, max 10 subplots
                break

            ax = fig.add_subplot(10, 1, subplot_idx)  # Use 10-row grid
            subplot_idx += 1

            for i, entry in enumerate(logfiles):
                if (
                    comp == "default"
                    or "component" not in df_newton[i][metric].index.names
                ):
                    dx_x = df_newton[i][metric].dropna()
                else:
                    dx_x = df_newton[i][metric].unstack("component")[comp].dropna()

                logfile_time_steps = dx_x.index.get_level_values("time_step").tolist()
                logfile_iteration_numbers = dx_x.index.get_level_values(
                    "iteration_number"
                ).tolist()
                logfile_pairs = list(zip(logfile_time_steps, logfile_iteration_numbers))

                x_positions = []
                y_values = []
                for pos, (ts, it) in enumerate(unique_combinations):
                    if (ts, it) in logfile_pairs:
                        idx = logfile_pairs.index((ts, it))
                        x_positions.append(pos)
                        y_values.append(dx_x.iloc[idx])
                    else:
                        x_positions.append(pos)
                        y_values.append(np.nan)

                # Label: show log file name
                ax.plot(
                    x_positions,
                    y_values,
                    label=entry,
                    marker="o",
                    markersize=3,
                )

            tick_labels = [f"{ts}\n{it}" for ts, it in unique_combinations]
            ax.set_xticks(range(len(tick_labels)))
            ax.set_xticklabels(tick_labels, rotation=0, fontsize=8)

            # Set y-scale: use log if all values are positive, otherwise linear
            all_values = []
            for i, entry in enumerate(logfiles):
                if (
                    comp == "default"
                    or "component" not in df_newton[i][metric].index.names
                ):
                    dx_x = df_newton[i][metric].dropna()
                else:
                    dx_x = df_newton[i][metric].unstack("component")[comp].dropna()
                all_values.extend(dx_x.values)

            if len(all_values) > 0 and all(
                v > 0 for v in all_values if not np.isnan(v)
            ):
                ax.set_yscale("log")
            else:
                ax.set_yscale("linear")

            ax.legend()

            # Set ylabel based on metric and component
            comp_label = f" (component {comp})"
            if metric == "residual_norm":
                ax.set_ylabel(r"$\|r\|$" + comp_label)
            else:  # dx_x
                ax.set_ylabel(r"$\|dx\|/\|x\|$" + comp_label)

            ax.set_xlabel("time step / iteration number")
            ax.grid(True, alpha=0.3, which="both")

    # Embed figure in Tkinter window
    canvas_tk = FigureCanvasTkAgg(fig, master=scrollable_frame)
    canvas_tk.draw()
    canvas_tk.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Add navigation toolbar for zoom, pan, and other features (fixed at bottom)
    # Create a frame for the toolbar that's NOT scrollable
    toolbar_frame = tk.Frame(root)
    toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)

    toolbar = NavigationToolbar2Tk(canvas_tk, toolbar_frame)
    toolbar.update()

    # Configure scrolling for cross-platform support
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(event):
        if event.num == 4:  # Scroll up
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:  # Scroll down
            canvas.yview_scroll(1, "units")

    # Bind mousewheel for different platforms
    canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows/Mac
    canvas.bind_all("<Button-4>", _on_mousewheel_linux)  # Linux scroll up
    canvas.bind_all("<Button-5>", _on_mousewheel_linux)  # Linux scroll down

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    root.mainloop()


create_scrollable_plot_window()

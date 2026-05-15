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
for df_log in df_logs:
    df_it.append(time_step_vs_iterations(df_log))
    df_ts.append(analysis_time_step(df_log))
    df_ts[-1] = df_ts[-1].loc[0]
    df_newton.append(analysis_convergence_newton_iteration(df_log))
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

    # Create matplotlib figure for all plots
    fig = Figure(figsize=(12, 6 * 6))  # 6 plots, each ~6 inches tall
    fig.suptitle("Log Comparison - All Metrics", fontsize=16, fontweight="bold")

    # Plot 1: Iteration number vs time step
    ax1 = fig.add_subplot(6, 1, 1)
    for i, entry in enumerate(logfiles):
        ax1.plot(df_it[i]["iteration_number"], label=entry)
    ax1.legend()
    ax1.set_xlabel("time step")
    ax1.set_ylabel("iteration number")
    ax1.set_title("Iteration Number vs Time Step")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative iteration number
    ax2 = fig.add_subplot(6, 1, 2)
    for i, entry in enumerate(logfiles):
        ax2.plot(df_it[i]["iteration_number"].cumsum(), label=entry)
    ax2.legend()
    ax2.set_xlabel("time step")
    ax2.set_ylabel("iteration number")
    ax2.set_title("Cumulative Iteration Number")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Step start time
    ax3 = fig.add_subplot(6, 1, 3)
    for i, entry in enumerate(logfiles):
        ax3.plot(df_it[i]["step_start_time"], label=entry)
    ax3.legend()
    ax3.set_xlabel("time step")
    ax3.set_ylabel("time / s")
    ax3.set_title("Step Start Time")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Step size
    ax4 = fig.add_subplot(6, 1, 4)
    for i, entry in enumerate(logfiles):
        ax4.plot(df_ts[i]["step_size"], label=entry)
    ax4.legend()
    ax4.set_xlabel("time step")
    ax4.set_ylabel("time step size / s")
    ax4.set_title("Step Size vs Time Step")
    ax4.grid(True, alpha=0.3)

    # Plot 5: Newton iteration dx_x for all components
    if len(df_newton) > 0:
        components = df_newton[0]["dx_x"].index.get_level_values("component").unique()
        n_components = len(components)

        if n_components > 0:
            # Find all unique time_step/iteration combinations
            all_time_steps = []
            all_iteration_numbers = []
            for i in range(len(logfiles)):
                for comp in components:
                    dx_x = df_newton[i]["dx_x"].unstack("component")[comp].dropna()
                    all_time_steps.extend(
                        dx_x.index.get_level_values("time_step").tolist()
                    )
                    all_iteration_numbers.extend(
                        dx_x.index.get_level_values("iteration_number").tolist()
                    )

            unique_combinations = sorted(
                set(zip(all_time_steps, all_iteration_numbers))
            )

            # Plot for each component (combine into remaining subplots)
            plots_per_component = (6 - 4) // n_components if n_components <= 2 else 1

            for comp_idx, comp in enumerate(components):
                if comp_idx + 5 <= 6:  # Stay within 6 subplots
                    ax = fig.add_subplot(6, 1, comp_idx + 5)
                    for i, entry in enumerate(logfiles):
                        dx_x = df_newton[i]["dx_x"].unstack("component")[comp].dropna()
                        logfile_time_steps = dx_x.index.get_level_values(
                            "time_step"
                        ).tolist()
                        logfile_iteration_numbers = dx_x.index.get_level_values(
                            "iteration_number"
                        ).tolist()
                        logfile_pairs = list(
                            zip(logfile_time_steps, logfile_iteration_numbers)
                        )

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

                        ax.plot(
                            x_positions, y_values, label=entry, marker="o", markersize=3
                        )

                    tick_labels = [f"{ts}\n{it}" for ts, it in unique_combinations]
                    ax.set_xticks(range(len(tick_labels)))
                    ax.set_xticklabels(tick_labels, rotation=0, fontsize=8)
                    ax.set_yscale("log")
                    ax.legend()
                    ax.set_ylabel(r"$dx_x$ (component {})".format(comp))
                    ax.set_xlabel("time step / iteration number")
                    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

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

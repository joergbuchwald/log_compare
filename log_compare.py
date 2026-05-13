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
for i, entry in enumerate(logfiles):
    plt.plot(df_it[i]["iteration_number"], label=entry)
plt.legend()
plt.xlabel("time step")
plt.ylabel("iteration number")
plt.show()
# %%
for i, entry in enumerate(logfiles):
    plt.plot(df_it[i]["step_start_time"], label=entry)
plt.legend()
plt.xlabel("time step")
plt.ylabel("time / s")
plt.show()
# %%
for i, entry in enumerate(logfiles):
    plt.plot(df_ts[i]["step_size"], label=entry)
plt.legend()
plt.xlabel("time step")
plt.ylabel("time step size / s")
plt.show()
# %%
# Plot dx_x for each component in subplots
components = df_newton[0]["dx_x"].index.get_level_values("component").unique()
n_components = len(components)

if n_components > 0:
    fig, axes = plt.subplots(
        n_components, 1, figsize=(14, 5 * n_components), sharex=True
    )

    # Handle case where there's only one component (axes is not an array)
    if n_components == 1:
        axes = [axes]

    for comp_idx, comp in enumerate(components):
        for i, entry in enumerate(logfiles):
            # Filter data for this component
            dx_x = df_newton[i]["dx_x"].unstack("component")[comp].dropna()

            # Get numeric x positions
            x_positions = range(len(dx_x))

            # Plot the data with log scale
            axes[comp_idx].plot(x_positions, dx_x.values, label=entry)

        # Set up two-row x-axis labels (time_step and iteration_number)
        dx_x_ref = df_newton[0]["dx_x"].unstack("component")[comp].dropna()
        time_steps = dx_x_ref.index.get_level_values("time_step").tolist()
        iteration_numbers = dx_x_ref.index.get_level_values("iteration_number").tolist()

        # Create two-row tick labels
        tick_labels = [f"{ts}\n{it}" for ts, it in zip(time_steps, iteration_numbers)]

        axes[comp_idx].set_xticks(range(len(tick_labels)))
        axes[comp_idx].set_xticklabels(tick_labels, rotation=0, fontsize=9)

        axes[comp_idx].set_yscale("log")
        axes[comp_idx].legend()
        axes[comp_idx].set_ylabel(r"$dx_x$ (component {})".format(comp))
        axes[comp_idx].grid(True, alpha=0.3, which="both")

    axes[-1].set_xlabel("time step / iteration number")
    plt.tight_layout()
    plt.show()

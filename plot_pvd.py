#!/usr/bin/python
# %%
"""
Scrollable Time Series Plot for Multiple PVD Files
===================================================

This script reads multiple PVD/XDMF files and plots time series data for
all variables at hardcoded observation points. Files are overlayed on the
same plot with different colors/styles.

Uses Tkinter for a scrollable window containing all matplotlib plots.

Usage:
    python plot_pvd.py file1.pvd file2.pvd file3.pvd -x 6.67179 -y -851.812 -z 0.0
"""

import argparse
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
import ogstools as ot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# %% [markdown]
# # **Parse command line arguments for PVD/XDMF files**

parser = argparse.ArgumentParser(
    description="Plot time series from multiple PVD/XDMF files at observation points."
)
parser.add_argument(
    "pvd_files",
    nargs="+",
    help="Paths to the PVD or XDMF files to plot (multiple files supported)",
)
parser.add_argument(
    "-x", type=float, required=True, help="X coordinate of observation point"
)
parser.add_argument(
    "-y", type=float, required=True, help="Y coordinate of observation point"
)
parser.add_argument(
    "-z", type=float, required=True, help="Z coordinate of observation point"
)
args = parser.parse_args()

# Construct observation point from command-line arguments
OBSERVATION_POINTS = np.array([[args.x, args.y, args.z]])

# %% [markdown]
# # **Load all mesh series files**

print(f"Loading {len(args.pvd_files)} file(s)...")
mesh_series_list = []
file_labels = []

for i, filepath in enumerate(args.pvd_files):
    ms = ot.MeshSeries(filepath).scale(time="a")
    mesh_series_list.append(ms)
    # Use filename as label
    label = filepath.split("/")[-1].split("\\")[-1]
    file_labels.append(label)
    print(f"  Loaded: {label}")

# %% [markdown]
# # **Probe observation points and collect data**

print(f"\nProbing {len(OBSERVATION_POINTS)} observation points...")

# Data structure: data_dict[variable][file_index] = {"time": ..., "data": ...}
data_dict = {}
all_variables = set()

for file_idx, ms in enumerate(mesh_series_list):
    probe = ms.probe(OBSERVATION_POINTS)
    # Get variable names (excluding coordinates and time)
    point_data_keys = list(probe.point_data.keys())
    variables = [
        var for var in point_data_keys if var.lower() not in ["x", "y", "z", "time"]
    ]

    for var_name in variables:
        all_variables.add(var_name)
        if var_name not in data_dict:
            data_dict[var_name] = {}
        # Store both time values and data for each file
        file_time_values = ms.timevalues
        file_data = ot.variables.Variable(var_name).transform(probe)
        data_dict[var_name][file_idx] = {
            "time": file_time_values,
            "data": file_data,
        }

all_variables = sorted(list(all_variables))
print(f"Found {len(all_variables)} unique variables:")
for v in all_variables:
    print(f"  - {v}")

# Build the list of all variable+component combinations
variable_components = []  # List of (var_name, component_idx or None, display_name)

for var_name in all_variables:
    first_file_idx = list(data_dict[var_name].keys())[0]
    first_data = data_dict[var_name][first_file_idx]["data"]

    # Data shape is (n_timesteps, n_points) for scalars or (n_timesteps, n_points, n_components) for vectors
    if len(first_data.shape) == 3 and first_data.shape[2] > 1:
        # Multi-component variable (e.g., velocity with X, Y, Z)
        n_components = first_data.shape[2]
        for comp_idx in range(n_components):
            comp_name = (
                ["X", "Y", "Z", "XX", "YY", "ZZ", "XY", "XZ", "YZ"][comp_idx]
                if comp_idx < 9
                else str(comp_idx)
            )
            display_name = f"{var_name} ({comp_name})"
            variable_components.append((var_name, comp_idx, display_name))
    else:
        # Scalar variable
        display_name = var_name
        variable_components.append((var_name, None, display_name))

print(f"\nTotal variable components to plot: {len(variable_components)}")

# Get time values from first file
time_values = mesh_series_list[0].timevalues
print(f"Number of timesteps: {len(time_values)}")

if len(variable_components) == 0:
    raise ValueError("No variables found in the provided files.")

# %% [markdown]
# # **Create scrollable Tkinter window with matplotlib plots**

# Color and style cycle for different files
colors = plt.cm.tab10.colors
styles = ["-"]  # All files use solid lines

num_points = len(OBSERVATION_POINTS)
num_var_components = len(variable_components)


def create_scrollable_plot_window():
    """Create a Tkinter window with scrollable matplotlib plots."""
    root = tk.Tk()
    root.title("PVD Time Series Viewer")
    root.geometry("1400x900")

    # Create main canvas with scrollbar
    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
    )

    # Enable mouse wheel scrolling (Linux uses Button-4/5, Windows/Mac use MouseWheel)
    def _on_mousewheel(event):
        # event.delta is 120 for scroll up, -120 for scroll down on Windows/Mac
        # On Linux, Button-4 is scroll up, Button-5 is scroll down
        if event.num == 4:  # Scroll up (Linux)
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:  # Scroll down (Linux)
            canvas.yview_scroll(1, "units")
        elif hasattr(event, "delta"):  # Windows/Mac
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # Bind mouse wheel for Linux (Button-4/5) and Windows/Mac (MouseWheel)
    canvas.bind_all("<Button-4>", _on_mousewheel)  # Scroll up (Linux)
    canvas.bind_all("<Button-5>", _on_mousewheel)  # Scroll down (Linux)
    canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows/Mac

    canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Bind canvas resize to update scrollable frame width
    def on_canvas_configure(event):
        canvas.itemconfig(canvas_window, width=event.width)

    canvas.bind("<Configure>", on_canvas_configure)

    # Pack canvas and scrollbar
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    # Create matplotlib figure for all plots with constrained layout
    # Each variable gets num_points rows, each row is ~5 inches tall (more vertically stretched)
    plot_height = 5  # inches per observation point row (more vertically stretched)
    total_plots = num_var_components * num_points
    fig = Figure(
        figsize=(14, total_plots * plot_height),
        constrained_layout=True,
    )
    fig.suptitle(
        "Time Series - All Variables",
        fontsize=14,
        fontweight="bold",
    )

    # Add matplotlib canvas to scrollable frame
    canvas_fig = FigureCanvasTkAgg(fig, master=scrollable_frame)
    canvas_fig.get_tk_widget().pack(fill="both", expand=True)

    # Add toolbar
    toolbar_frame = ttk.Frame(scrollable_frame)
    toolbar_frame.pack(fill="x", side="bottom")
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

    toolbar = NavigationToolbar2Tk(canvas_fig, toolbar_frame)
    toolbar.update()

    # Create axes for each variable
    axes_list = []
    for var_idx, (var_name, comp_idx, display_name) in enumerate(variable_components):
        # Create subplots for this variable's observation points
        for point_idx in range(num_points):
            # Calculate subplot position (sharex between all)
            row = total_plots - 1 - (var_idx * num_points + point_idx)
            ax = fig.add_subplot(
                total_plots,
                1,
                row + 1,
                sharex=axes_list[0] if axes_list else None,
            )
            axes_list.append(ax)

            # Plot data from all files for this observation point
            file_lines = []  # Track lines for legend
            file_labels_used = []  # Track labels for legend
            for file_idx, label in enumerate(file_labels):
                if var_name not in data_dict or file_idx not in data_dict[var_name]:
                    continue

                file_info = data_dict[var_name][file_idx]
                file_time = file_info["time"]
                var_data = file_info["data"]

                # Select appropriate component
                if comp_idx is not None and len(var_data.shape) == 3:
                    # Vector component selected
                    y_vals = var_data[:, point_idx, comp_idx]
                else:
                    # Scalar - just get the values for this point
                    if len(var_data.shape) == 3:
                        y_vals = var_data[:, point_idx, 0]
                    else:
                        y_vals = var_data[:, point_idx]

                color = colors[file_idx % len(colors)]
                style = styles[file_idx % len(styles)]
                line = ax.plot(
                    file_time,
                    y_vals,
                    color=color,
                    linestyle=style,
                    linewidth=1.5,
                    label=label,
                )
                file_lines.extend(line)
                file_labels_used.append(label)

            # Add y-label showing variable name (rotated along y-axis)
            if point_idx == 0:
                # First subplot: variable name along y-axis
                ax.set_ylabel(
                    display_name,
                    fontsize=10,
                    fontweight="bold",
                    rotation=90,
                    labelpad=15,
                )
            else:
                # Other subplots: observation point number
                ax.set_ylabel(f"Pt {point_idx}", fontsize=8)

            ax.grid(True, linestyle="--", alpha=0.5)
            ax.tick_params(axis="both", which="both", labelsize=7)

            # Add legend to each subplot
            ax.legend(
                loc="upper right",
                fontsize=7,
                framealpha=0.9,
                ncol=1 if len(file_labels_used) > 2 else 2,
            )

    # Set xlabel on bottom axis
    if axes_list:
        axes_list[-1].set_xlabel("Time", fontsize=9)

    # Tight layout adjustment
    fig.tight_layout(rect=[0.05, 0.03, 0.95, 0.97])

    # Draw the canvas
    canvas_fig.draw()

    return root


# Create and run the window
root = create_scrollable_plot_window()

print("\n" + "=" * 60)
print("A window will open with all variables plotted.")
print("Scroll using the scrollbar on the right side.")
print("=" * 60)

root.mainloop()

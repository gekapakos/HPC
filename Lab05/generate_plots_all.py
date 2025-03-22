import matplotlib.pyplot as plt

# Data
labels = [
    "CPU_Original", "OMP_Accelerated", "OMP_Accelerated_2", "CUDA_0_no_optimization", 
    "CUDA_1_registers", "CUDA_2_SoA", "CUDA_3_SoA_float3", "CUDA_4_SoA_float4", 
    "CUDA_5_tiling_shared", "CUDA_6_unroll", "CUDA_7_fast_math", "CUDA_8_final"
]
execution_times = [
    3.090e+01, 1.191e+00, 1.265e+00, 5.334e-01, 5.382e-01, 5.679e-01, 5.441e-01, 5.218e-01, 
    4.369e-01, 4.363e-01, 1.946e-01, 1.796e-01
]

std_devs = [
    3.11E-01, 1.79E-02, 3.32E-03, 2.72E-03, 2.68E-03, 1.578E-03, 2.101E-03, 
    1.787E-03, 2.450E-03, 1.697E-03, 5.78E-04, 6.79E-04
]

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 8), dpi=300)  # High resolution
bar_width = 0.6
x = range(len(labels))

# Plot bars for execution time
bars = ax1.bar(x, execution_times, width=bar_width, color="lightblue", edgecolor="black", label="Execution Time")

# Add values on top of bars in scientific notation
for bar, value in zip(bars, execution_times):
    ax1.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2e}",
        ha='center', va='bottom', fontsize=9, color="black"
    )

# Customize the primary y-axis
ax1.set_ylabel("Execution Time (s)", fontsize=12)
ax1.set_xlabel("Optimization Path", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)

# Create a secondary y-axis for standard deviation
ax2 = ax1.twinx()
ax2.plot(x, std_devs, color="green", marker="o", label="Standard Deviation", linestyle="-")

# Customize the secondary y-axis
ax2.set_ylabel("Standard Deviation (s)", fontsize=12)

# Add legends
bar_legend = ax1.legend(loc="upper left", fontsize=10)
std_dev_legend = ax2.legend(loc="upper right", fontsize=10)

# Add title
plt.title("Execution Time for Different Optimization for Bodies 131072", fontsize=14)

# Save the plot
output_file = "optimization_paths_execution_time_with_std_dev.png"
plt.tight_layout()
plt.savefig(output_file)
print(f"Plot saved: {output_file}")
plt.close()

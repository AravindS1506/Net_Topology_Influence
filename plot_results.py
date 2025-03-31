import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 20,
    'pdf.fonttype': 42,  # Use Type 42 (TrueType)
    'ps.fonttype': 42    # Use Type 42 (TrueType)
    
})
plt.rcParams['xtick.major.pad'] = 15
# Numerical x values, making "Initial" at x=0
x_values = [0, 1.25, 2.5, 3.75, 5]
x_labels = ["Initial Graph", "(WIN,VIC,\nROM,0.32)", "(LOU,JOH,\nBER,0.05)", "(LOU,ALB,\nAMB,0.24)",
            "(BONI,BER,\nPET,0.96)"]
s1_values = [0.3673, 0.5181, 0.7369, 0.8195,0.8988]
s2_values = [1 - s1 for s1 in s1_values]
plt.figure(figsize=(12, 6))
plt.plot(x_values, s1_values, marker='o', linestyle='-', color='r', label="Influence of HUGH")

# Plot s2_values in blue
plt.plot(x_values, s2_values, marker='s', linestyle='-', color='b', label="Influence of PETER")

# Annotate each point with its value

plt.text(x_values[0]+0.02, s1_values[0]+0.05, f'{s1_values[0]:.3f}', ha='center', va='bottom', color='r')
plt.text(x_values[0]+0.02, s2_values[0]-0.06, f'{s2_values[0]:.3f}', ha='center', va='top', color='b')

plt.text(x_values[1], s1_values[1]+0.04, f'{s1_values[1]:.3f}', ha='center', va='bottom', color='r')
plt.text(x_values[1], s2_values[1]-0.04, f'{s2_values[1]:.3f}', ha='center', va='top', color='b')

plt.text(x_values[2], s1_values[2]-0.12, f'{s1_values[2]:.3f}', ha='center', va='bottom', color='r')
plt.text(x_values[2], s2_values[2]+0.1, f'{s2_values[2]:.3f}', ha='center', va='top', color='b')

plt.text(x_values[3]-0.08, s1_values[3]+0.023, f'{s1_values[3]:.3f}', ha='center', va='bottom', color='r')
plt.text(x_values[3]-0.11, s2_values[3]-0.027, f'{s2_values[3]:.3f}', ha='center', va='top', color='b')

plt.text(x_values[4], s1_values[4]-0.1, f'{s1_values[4]:.3f}', ha='center', va='bottom', color='r')
plt.text(x_values[4], s2_values[4]+0.1, f'{s2_values[4]:.3f}', ha='center', va='top', color='b')

# plt.text(x_values[5]-0.2, s1_values[5]+0.02, f'{s1_values[5]:.3f}', ha='center', va='bottom', color='r')
# plt.text(x_values[5]-0.3, s2_values[5]-0.02, f'{s2_values[5]:.3f}', ha='center', va='top', color='b')

# plt.text(x_values[6]-0.05, s1_values[6]-0.1, f'{s1_values[6]:.3f}', ha='center', va='bottom', color='r')
# plt.text(x_values[6]-0.05, s2_values[6]+0.1, f'{s2_values[6]:.3f}', ha='center', va='top', color='b')

# Set X-axis tick positions and labels
plt.xticks(x_values, x_labels, rotation=0)
# plt.subplots_adjust(bottom=0.25, top=0.9)   # Increase space at botto
# Set y-axis range from 0 to 1
plt.ylim(0, 1)

# Draw horizontal line at y=0.5
plt.axhline(y=0.5, color='black', linestyle='dotted')
plt.text(-0.3, 0.5, "0.5", ha='right', va='center', color='black',fontsize=18)
plt.xlabel("Edge modifications (a,b,d,weight)",labelpad=10)
plt.ylabel("Influence Centrality", labelpad=15)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.1, alpha=0.1)
plt.tight_layout()
plt.savefig("influence_plot.eps", format='eps', bbox_inches='tight')
plt.show()

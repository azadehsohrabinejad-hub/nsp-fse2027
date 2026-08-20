import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_trajectories():
    report_path = r"D:\raz\razieh\reports\pilot_drift_analysis.json"
    
    with open(report_path, 'r') as f:
        results = json.load(f)
        
    # Select a few interesting trajectories with high drift to plot
    # Let's pick trajectories 2, 12, and 19 (indices 1, 11, 18)
    indices_to_plot = [1, 11, 18]
    
    fig, axes = plt.subplots(len(indices_to_plot), 1, figsize=(10, 12), sharex=True)
    
    for i, idx in enumerate(indices_to_plot):
        if idx >= len(results):
            continue
            
        traj = results[idx]
        rounds = list(range(1, traj["length_T"] + 1))
        
        p_t = traj["repair_progress"]
        s_t = traj["semantic_alignment"]
        
        ax = axes[i]
        
        # Plot Repair Progress (Blue)
        ax.plot(rounds, p_t, label='Repair Progress (p_t)', color='blue', marker='o')
        # Plot Semantic Alignment (Red)
        ax.plot(rounds, s_t, label='Semantic Alignment (s_t)', color='red', marker='x', linestyle='--')
        
        ax.set_title(f"Trajectory {idx+1} - Total Deceptive Drift: {traj['total_deceptive_drift']:.4f}")
        ax.set_ylabel("Latent State Value")
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='best')
        
    axes[-1].set_xlabel("Repair Round (t)")
    plt.suptitle("NSP Hidden State Evolution: Deceptive Semantic Drift", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    output_path = r"D:\raz\razieh\reports\Figure_6_Hidden_State_Evolution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Figure 6 saved successfully to: {output_path}")

if __name__ == "__main__":
    plot_trajectories()
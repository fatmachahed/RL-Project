import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ActorCritic
from src.generate_graphs import *

# =========================================================
# MAIN
# =========================================================
def main():
    PROJECT_ROOT = Path.cwd().parent  
    plots_dir = create_plots_directory(PROJECT_ROOT)
    
    print(" Chargement des données...")
    data = load_all_data("results")
    
    if not data:
        print("Aucune donnée trouvée dans 'results/'")
        return
    
    print(f"✅ {len(data)} agents trouvés:")
    for agent, runs in data.items():
        print(f"   • {agent}: {len(runs)} runs")
    
    print(f"\n📊 Génération des visualisations dans '{plots_dir}/'...")
    
    try:
        plot_learning_curves(data, plots_dir)
        plot_training_rewards(data, plots_dir)
        plot_loss_dynamics(data, plots_dir)
        plot_value_evolution(data, plots_dir)
        plot_trajectory_values(data, plots_dir)
        plot_entropy_evolution(data, plots_dir)
        plot_performance_comparison(data, plots_dir)
        plot_training_stability(data, plots_dir)
        plot_correlation_heatmap(data, plots_dir)
        plot_convergence_speed(data, plots_dir)
        plot_final_rewards_per_agent(data, plots_dir)

    except Exception as e:
        print(f"Erreur lors de la génération: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Toutes les visualisations ont été générées!")
    print("\n Fichiers créés (11 graphes):")
    print("   1_learning_curves.png          - Eval rewards (LE PLUS IMPORTANT)")
    print("   2_training_rewards.png         - Train rewards evolution")
    print("   3_loss_dynamics.png            - Actor & Critic losses")
    print("   4_value_evolution.png          - Value function (all + zoom)")
    print("   5_trajectory_values.png        - V(s) on trajectories")
    print("   6_entropy_evolution.png        - Exploration behavior (all + zoom)")
    print("   7_performance_comparison.png   - Final performance boxplots")
    print("   8_training_stability.png       - Stability metrics")
    print("   9_correlation_heatmap.png     - Metrics correlations")
    print("   10_convergence_speed.png       - Speed (steps vs updates)")
    print("   11_final_rewards_per_agent.png - Final rewards per agent")
    print("\n Analyse terminée!")


if __name__ == "__main__":
    main()
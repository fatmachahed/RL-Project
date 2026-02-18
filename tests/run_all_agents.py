from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')
from src.train_multi_seed import train

# =========================================================
# MAIN
# =========================================================
def main():
    AGENTS = ["agent0","agent1","agent2","agent3","agent4"]
    SEEDS = [42, 123, 999]  
    CONFIG_DIR = Path("configs")
    
    Path("results").mkdir(exist_ok=True)
    
    if not CONFIG_DIR.exists():
        print("Dossier 'configs' introuvable!")
        return
    
    for agent in AGENTS:
        cfg_path = CONFIG_DIR / f"{agent}.yaml"
        
        if not cfg_path.exists():
            print(f"Fichier ignoré: {cfg_path}")
            continue
        
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            required_fields = ['env_name', 'K', 'n_steps', 'lr_actor', 'lr_critic', 
                             'gamma', 'max_steps', 'entropy_coef', 'eval_freq']
            
            missing = [f for f in required_fields if f not in config]
            if missing:
                print(f"Champs manquants dans {agent}.yaml: {missing}")
                continue
            
            print(f"\n{'='*70}")
            print(f" TRAINING {agent.upper()}")
            print(f"{'='*70}")
            print(f"K={config['K']}, n={config['n_steps']}, "
                  f"stochastic={config.get('stochastic_rewards', False)}")
            print(f"lr_actor={config['lr_actor']}, lr_critic={config['lr_critic']}")
            
            seeds_ok = 0
            for seed in SEEDS:
                print(f"\n Seed {seed}...")
                try:
                    train(config, seed, agent)
                    seeds_ok += 1
                except Exception as e:
                    print(f"Erreur: {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"\n{agent}: {seeds_ok}/{len(SEEDS)} seeds OK")
                    
        except Exception as e:
            print(f"Erreur {agent}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("  ✅ TRAINING TERMINÉ")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
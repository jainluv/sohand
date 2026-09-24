import argparse
import os
from pathlib import Path
import time

try:
    from stable_baselines3 import PPO  
    # NOTE: If your model was trained using a different algorithm (like SAC, TD3, or DDPG), 
    # replace 'PPO' above with your algorithm, e.g.: from stable_baselines3 import SAC
except ImportError:
    raise ImportError("Stable Baselines3 is not installed. Run: pip install stable-baselines3")

def main():
    parser = argparse.ArgumentParser(description="View and evaluate a trained So-Hand RL model.")
    parser.add_argument(
        "--model",
        type=str,
        default=r"C:\Users\luvja\Desktop\so-hand\sohand\runs\spin\models\best_model.zip",
        help="Absolute or relative path to the trained model .zip file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run and visualize"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="Whether to render the environment visually"
    )

    args = parser.parse_args()

    # 1. Robust path normalization and validation
    model_path = Path(args.model).resolve()
    print(f"[INFO] Resolving model path: {model_path}")

    if not model_path.exists():
        print(f"[ERROR] Model file not found at: {model_path}")
        print("[TIP] Verify that the file exists and that you have correct read permissions.")
        return

    # 2. Load the trained model
    print(f"[INFO] Loading model from {model_path}...")
    try:
        # Change PPO.load(...) to your specific algorithm class if trained with SAC/TD3/etc.
        model = PPO.load(str(model_path))
        print("[SUCCESS] Model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load the model zip file: {e}")
        return

    # 3. Set up the Environment
    try:
        import gymnasium as gym
        
        # If your repository registers a custom environment package, import it here:
        # import sohand  # uncomment if 'sohand' registers its env on import
        
        env_id = "sohand-v0"  # Update this string if your environment uses a different registration ID
        print(f"[INFO] Initializing environment: {env_id}")
        
        env = gym.make(env_id, render_mode="human" if args.render else None)
    except Exception as e:
        print(f"[WARNING] Could not automatically initialize gym environment: {e}")
        print("[INFO] Please verify your environment registration ID or custom gym setup.")
        return

    # 4. Evaluation & Visualization Loop
    print(f"\n[INFO] Starting evaluation for {args.episodes} episodes...")
    try:
        for episode in range(1, args.episodes + 1):
            obs, info = env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            step_count = 0

            print(f"\n--- Running Episode {episode}/{args.episodes} ---")
            while not (done or truncated):
                # Predict the optimal action deterministically
                action, _states = model.predict(obs, deterministic=True)
                
                # Step through the environment
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1

                if args.render:
                    env.render()
                
                # Optional: Uncomment if simulation runs too fast
                # time.sleep(0.01)

            print(f"Episode {episode} finished in {step_count} steps. Total Reward: {total_reward:.4f}")

    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted manually by user.")
    except Exception as e:
        print(f"[ERROR] An error occurred during the simulation loop: {e}")
    finally:
        env.close()
        print("[INFO] Environment closed cleanly. Execution complete.")

if __name__ == "__main__":
    main()
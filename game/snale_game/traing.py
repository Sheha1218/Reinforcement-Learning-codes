from logging import log
from stable_baselines3 import PPO
from snake_env import SnakeEnv
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def train_snake(timesteps=100000, render=False):


    logging.info("Training Snake with PPO")
    logging.info(f"Total timesteps: {timesteps}")
    logging.info("-" * 40)

    render_mode = "human" if render else None
    env = SnakeEnv(render_mode=render_mode)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

   
    logging.info("Starting training...")
    model.learn(total_timesteps=timesteps)

    
    model.save("snake_model")
    logging.info("Model saved as 'snake_model'")

    env.close()
    return model

def play_trained_model(model_path="snake_model", episodes=5):
    logging.info(f"Loading model: {model_path}")
    env = SnakeEnv(render_mode="human")
    model = PPO.load(model_path, env=env)

    logging.info(f"Watching trained agent play {episodes} episodes...")
    logging.info("Close the window to stop early")

    scores = []
    for episode in range(episodes):
        obs, info = env.reset()
        done = False

        logging.info(f"Episode {episode + 1}:")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        score = info.get('score', 0)
        scores.append(score)
        logging.info(f"Score: {score}")

    env.close()

    logging.info(f"\nResults:")
    logging.info(f"Average Score: {sum(scores)/len(scores):.2f}")
    logging.info(f"Best Score: {max(scores)}")

    return scores

def main():
    import sys

    if len(sys.argv) == 1:
        train_snake()
    elif sys.argv[1] == "train":
        timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
        render = "--render" in sys.argv
        train_snake(timesteps, render)
    elif sys.argv[1] == "play":
        model_path = sys.argv[2] if len(sys.argv) > 2 else "snake_model"
        episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        play_trained_model(model_path, episodes)
    else:
        logging.info("Usage:")
        logging.info("  python train_snake.py                    # Train with defaults")
        logging.info("  python train_snake.py train 200000       # Train for 200k steps")
        logging.info("  python train_snake.py train 50000 --render # Train with rendering")
        logging.info("  python train_snake.py play               # Watch trained model")
        logging.info("  python train_snake.py play snake_model 3 # Watch model play 3 games")

if __name__ == "__main__":
    main()
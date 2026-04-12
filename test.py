import gymnasium as gym
import mani_skill2.envs  # Регистрация окружений ManiSkill2
import os
import torch    

# Убедитесь, что используется EGL для headless-рендеринга
os.environ['MUJOCO_GL'] = 'egl'

try:
    print(gym.envs.registry.keys())
    # В MS2 v0.5.0+ используется PickCube-v0
    env = gym.make("PickCube-v0", obs_mode="state")
    obs, _ = env.reset()
    print("ManiSkill2 check successful!")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Device:", torch.cuda.get_device_name(0))
    print("Observation shape:", obs.shape)
    env.close()
except Exception as e:
    print(f"Error: {e}")
    print("\nAvailable ManiSkill2 environments:")
    # Вывод списка всех зарегистрированных ID для проверки правильного имени
    envs = [e for e in gym.envs.registry.keys() if "PickCube" in e]
    print(envs)

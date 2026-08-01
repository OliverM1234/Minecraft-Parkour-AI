import minedojo

print("Initializing MineDojo environment...")
env = minedojo.make(
    task_id="open-ended",
    image_size=(160, 256)
)

obs = env.reset()
print("Success! Environment reset complete. Observation keys:", obs.keys())
env.close()
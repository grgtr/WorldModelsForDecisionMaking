for env in dmc_cartpole_swingup dmc_walker_walk maniskill_LiftCube-v0; do
  for seed in 1 2 3; do
    conda run --no-capture-output -n .venv_world_models python -u src/analyze_latents.py \
      --explicit_ckpt logs/explicit/${env}/seed_${seed}/checkpoints/checkpoint_0200000.pt \
      --implicit_ckpt logs/implicit/${env}/seed_${seed}/checkpoints/checkpoint_0200000.pt \
      --env ${env} --seed ${seed} \
      --output analysis_v2/${env}/seed_${seed}/ \
      --skip_env_collection
  done
done
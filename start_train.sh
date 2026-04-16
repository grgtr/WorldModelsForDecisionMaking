for env in dmc_cartpole_swingup dmc_walker_walk; do
  for seed in 1 2 3; do
    conda run --no-capture-output -n .venv_world_models python -u src/train.py --agent explicit --env $env --seed $seed --total_steps 200000
    conda run --no-capture-output -n .venv_world_models python -u src/train.py --agent implicit --env $env --seed $seed --total_steps 200000
  done
done



for env in maniskill_LiftCube-v0; do
  for seed in 1 2; do
    conda run --no-capture-output -n .venv_world_models python src/train.py --agent explicit --env $env --seed $seed
  done
done


# Bot dev workflow

Run elf-sims solely for bot development, running everything else in docker.

That includes the data dashboard. If you want to work on the dashboard, use the [DATADEV.md](DATADEV.md) workflow.

This workflow uses the following packages:
[elf-simulations](https://github.com/delvtech/elf-simulations/blob/main/INSTALL.md),
[agent0](https://github.com/delvtech/elf-simulations/tree/main/lib/agent0),
[infra](https://github.com/delvtech/infra/blob/main/README.md)

1. have pyenv and **pyenv-virtualenv installed** ([pyenv instructions](https://github.com/pyenv/pyenv#installation))
2. clone repo
3. set up virtualenv:
```zsh
pyenv install 3.10
pyenv global 3.10
pyenv virtualenv mihaipy
cd <repo_location>
pyenv local mihaipy
```
4. install elf-sims:
```zsh
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```
5. check out version of [infra repo](https://github.com/delvtech/infra/) to use (check main and PRs)
    - for example, on August 8 2023, main has `ELFPY_TAG=0.3.2` but I want to use latest elf-sims `ELFPY_TAG=0.4.1` so I check out [PR #66](https://github.com/delvtech/infra/pull/66) which has it
6. check out [hyperdrive repo](https://github.com/delvtech/hyperdrive/) in sym-linked folder (as per instructions)
7. set `ETH_PORT=8546` in `env/env.ports` to avoid port conflict
8. create `.env` with ``./setup_env.sh --devnet --ports --postgres --data``
9. docker login if you need to (should persist after first time)
10. run `docker compose down -v` (set to `dcd` alias) to ensure nothing is running in docker
11. run `docker compose up --pull always` (set to `dcu` alias) to run selected docker services
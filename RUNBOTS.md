# Run bots workflow

Lets you run bots.

1. have python 3.10 installed
2. **optional** set up a virtual environment like [pyenv](https://github.com/pyenv/pyenv#installation) or [venv](https://docs.python.org/3/library/venv.html)
3. install [infra repo](https://github.com/delvtech/infra/), run with `./setup_env.sh --devnet --ports --postgres --data`
4. install bots `pip install git+https://github.com/delvtech/ro-bots.git`
5. run bots `python -c "import ro_bots; ro_bots.run()"`

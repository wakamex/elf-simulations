cp .logging/anvil_crash.json .logging/anvil_crash_live.json
anvil --tracing --code-size-limit=9999999999999999 --state .logging/anvil_crash_live.json --accounts 1

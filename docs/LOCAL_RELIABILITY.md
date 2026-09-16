# NATIP Local Reliability

NATIP is a local Streamlit desktop-style app. It is not a permanent server unless
you start it or macOS starts it for you.

## Fastest way to open NATIP

Double-click:

```text
NATIP.command
```

This starts NATIP and opens:

```text
http://127.0.0.1:8501
```

## Terminal commands

From the NATIP folder:

```bash
scripts/start_natip.sh
scripts/status_natip.sh
scripts/stop_natip.sh
```

Logs are written to:

```text
logs/streamlit.log
```

## Why it stops after a few days

The old workflow started Streamlit as a temporary terminal/Codex process. That
process can disappear after sleep, restart, logout, terminal close, or a Codex
session ending.

The new launcher starts NATIP in a repeatable way and checks that Streamlit is
pinned to the locally stable version.

## Auto-start on login

A LaunchAgent template exists at:

```text
config/com.natip.local.plist
```

macOS may block background LaunchAgents from reading projects stored under
Desktop unless Terminal/bash has the required Privacy access. If the LaunchAgent
log shows `Operation not permitted`, either:

1. Use `NATIP.command`, or
2. Grant Terminal/bash Full Disk Access in macOS Settings, or
3. Move the NATIP project outside Desktop and update the plist paths.

This is a macOS privacy restriction, not a NATIP application failure.

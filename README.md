# NBClaw

No Bullshit Claw: a small daemon that puts a [Swival](https://swival.dev/) agent  n the other end of a Signal conversation.

You text it. It does the work and texts back. You can also tell it to do things on a schedule ("every weekday at 9, summarize the git log") and cancel those later.

## Dependencies

- Python 3.13+ and [uv](https://docs.astral.sh/uv/).
- A running model. Anything Swival supports works; the quickest is to use [LM Studio](https://lmstudio.ai/) with a tool-calling model loaded.
- `signal-cli` registered to a phone number and running as an HTTP daemon:

  ```sh
  signal-cli --account XXXXXXXXX daemon --http 127.0.0.1:3080
  ```

## Quick start

```sh
cd nbclaw
uv sync

uv run nbclaw \
    --allow YOURPHONE \
    --model ornith-1.0-9b \
    --notify YOURPHONE
```

`--allow` is the number that's permitted to drive the agent (your phone).
`--notify` is optional and just sends a "nbclaw is online" message on start.

Now message the signal-cli number from your phone:

```text
you   :  list the python files in ~/src and count them
nbclaw:  There are 42 .py files under ~/src. ...
```

## Authorization (read this)

By default the agent runs in **autonomous** mode: it can run shell commands and read or edit files anywhere the account it runs as can reach. The workspace is just where it starts, not a fence. But it means **anyone on the allowlist effectively has a shell on this machine**.

- Always set `--allow` to the specific numbers you trust. With no allowlist set, every incoming message is ignored (fail closed).
- `--allow-all` exists for testing only. Don't use it on a machine you care about.
- `--safe` makes the agent read-only: no shell commands, no file edits. Good for a "just answer questions" bot.
- In a group chat, authorization is checked against the **sender**, but the reply goes to the **whole group**. So one allowed member can make the bot post agent output to everyone in that group. Keep the allowlist to people you trust with that, or only message the bot in 1:1 chats / Note to Self.

## Commands

Send these as Signal messages. Anything not starting with `/` goes to the agent.

| Command                 | What it does                          |
| ----------------------- | ------------------------------------- |
| `/help`                 | List the commands.                    |
| `/status`               | Model, mode, uptime, number of crons. |
| `/clear`                | Forget this conversation's context.   |
| `/cron <plain English>` | Schedule a task, described naturally. |
| `/cron list`            | Show scheduled tasks.                 |
| `/cron del <name>`      | Cancel a scheduled task.              |
| `/cron run <name>`      | Run a scheduled task right now.       |

### Scheduling

Just say it after `/cron` in plain English. The model works out the timing and gives the task a short name:

```text
/cron every weekday at 9am summarize my git log in ~/src/app
/cron remind me to stretch every 2 hours
/cron tomorrow at 8am say good morning
```

Both recurring schedules and one-time reminders ("in 10 minutes…", "tomorrow at 8am…") are understood; one-time jobs delete themselves after they fire.

Results are delivered to the conversation that created the cron, prefixed with its name. Crons run as independent one-shots, so they never pollute your chat's context.

Use `/cron list` to see names, then `/cron del <name>` to cancel.

If you'd rather be exact, the power-user form takes a literal schedule:

```text
/cron add standup 0 9 * * 1-5 | summarize today's commits in ~/src/myrepo
```

where the schedule is a 5-field cron expression, `@every 30m` (`30s`/`5m`/`2h`/`1d`), or `@hourly` / `@daily` / `@weekly` / `@monthly`.

## Configuration file

Flags cover the common cases. For anything else, point `--config` at a TOML file.

Top-level keys mirror the settings; a `[swival]` table is passed straight through to `swival.Session`, so the full agent is configurable.

```toml
# nbclaw.toml
signal_url = "http://127.0.0.1:3080"
allow = ["YOURPHONE"]
notify = "YOURPHONE"

provider = "lmstudio"
model = "ornith-1.0-9b"
# base_url = "http://127.0.0.1:1234"   # only if not the provider default
max_turns = 60

state_dir = "~/.nbclaw"

# MCP servers, in swival's format.
[mcp_servers.fetch]
command = "uvx"
args = ["mcp-server-fetch"]

# Anything here is forwarded verbatim to swival.Session.
[swival]
temperature = 0.2
reasoning_effort = "medium"
```

Run it:

```sh
uv run nbclaw --config nbclaw.toml
```

CLI flags override the file.

## Running 24/7

The agent is meant to stay up.

On macOS, a launchd job keeps it alive across logouts and reboots. A template is in `deploy/com.nbclaw.daemon.plist`; edit the paths and numbers, then:

```sh
cp deploy/com.nbclaw.daemon.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.nbclaw.daemon.plist
```

On Linux, a user systemd unit does the same job; the plist documents the same command line.

## State

Everything lives under `state_dir` (default `~/.nbclaw`):

- `crons.json` — scheduled tasks, written atomically.
- `workspace/` — the agent's working directory (its `base_dir`).

## Environment variables

- `NBCLAW_LOG` sets the log level (default `INFO`; try `DEBUG`).

## But why this since there's already XYZ?

NBClaw uses Swival as a Python library, so the CLI isn't required. This lets it work well even with small, local models and short context windows. No large models needed.

More importantly, NBClaw is ridiculously lightweight and incredibly easy to install and use.

No bloat. It intentionally ships with a minimal set of tools, but it gets the job done. And if you need more, you can extend it with skills, MCP servers, and whatever else fits your workflow.

It may not be for you, but this is the minimal claw-style agent I always wanted.

#!/usr/bin/env bash
# Production deploy script. Runs ON THE SERVER, streamed over SSH by
# .github/workflows/deploy.yml:
#
#   ssh prod 'REF=<git ref> ROOT=<checkout path> bash -s' < deploy/remote_deploy.sh
#
# Required environment: REF (branch, tag or full commit SHA to deploy) and ROOT (the
# git checkout holding docker-compose.yml, .env and data/). Optional: HEALTH_TIMEOUT,
# seconds to wait for the bot to come up (default 90).
#
# What it does: refuses to run over locally modified tracked files, updates the
# checkout to REF, installs the .env the workflow uploaded as .env.new (the old one is
# kept as .env.bak), rebuilds and restarts the compose stack and waits until the bot
# container is running and has logged "is running". Nothing here touches data/.
#
# Everything lives in functions and the last line calls main: the script arrives on
# stdin, and bash reads a function definition completely before running anything, so
# no command can swallow the rest of the script by reading stdin. main also gets
# stdin redirected from /dev/null for the same reason.
set -euo pipefail

log() { printf '\n==> %s\n' "$*"; }

bot_container() { docker compose ps -q -a bot 2>/dev/null | head -n 1; }

container_status() { docker inspect -f '{{.State.Status}}' "$1" 2>/dev/null || echo missing; }

container_started_at() { docker inspect -f '{{.State.StartedAt}}' "$1" 2>/dev/null || echo unknown; }

# Wait up to $1 seconds until the bot container is running and logged "is running"
# since its current start (aiogram logs "Bot @name is running" on startup), then
# make sure it is still the same running container 5 s later, so a crash right after
# the start line is not reported as success.
wait_for_bot() {
  local timeout=$1 deadline cid started logs
  deadline=$((SECONDS + timeout))
  while [ "$SECONDS" -lt "$deadline" ]; do
    cid=$(bot_container)
    if [ -n "$cid" ] && [ "$(container_status "$cid")" = running ]; then
      started=$(container_started_at "$cid")
      logs=$(docker logs --since "$started" "$cid" 2>&1 || true)
      if [[ "$logs" == *"is running"* ]]; then
        sleep 5
        if [ "$(container_status "$cid")" = running ] && [ "$(container_started_at "$cid")" = "$started" ]; then
          return 0
        fi
      fi
    fi
    sleep 5
  done
  return 1
}

refuse_if_dirty() {
  local dirty
  dirty=$(git status --porcelain --untracked-files=no)
  [ -z "$dirty" ] && return 0
  {
    echo "ERROR: tracked files are modified on the server; refusing to deploy over them:"
    echo "$dirty"
    echo
    echo "Fix it on the server first, then re-run the workflow:"
    echo "  cd $ROOT && git diff          # see what changed"
    echo "  git stash                     # set the change aside, or"
    echo "  git checkout -- <file>        # drop it, or commit and push it"
    echo "(.env and data/ are ignored by git and never part of this check.)"
  } >&2
  exit 1
}

checkout_ref() {
  git fetch --prune --tags origin
  if git show-ref --verify --quiet "refs/remotes/origin/$REF"; then
    # A branch: move (or create) the local branch to the remote tip.
    if git show-ref --verify --quiet "refs/heads/$REF"; then
      local ahead
      ahead=$(git rev-list --count "origin/$REF..$REF")
      if [ "$ahead" -gt 0 ]; then
        echo "WARNING: local branch $REF had $ahead unpushed commit(s) at $(git rev-parse --short "$REF"); they are left behind."
      fi
    fi
    git checkout -q -B "$REF" "origin/$REF"
  else
    # A tag or a commit SHA: fetch it explicitly when unknown, then check it out detached.
    git rev-parse --verify --quiet "${REF}^{commit}" >/dev/null || git fetch origin "$REF"
    git checkout -q --detach "$REF"
  fi
}

install_env_file() {
  if [ ! -f .env.new ]; then
    log "No .env.new was uploaded; keeping the current .env"
    return 0
  fi
  if [ -f .env ]; then
    cp -p .env .env.bak
    chmod 600 .env.bak
  fi
  mv -f .env.new .env
  chmod 600 .env
  log "Installed the new .env ($(wc -l < .env) lines); the previous one is kept as .env.bak"
}

main() {
  : "${REF:?REF (git ref to deploy) is not set}"
  : "${ROOT:?ROOT (checkout path on the server) is not set}"
  local health_timeout="${HEALTH_TIMEOUT:-90}" previous

  export BUILDKIT_PROGRESS=plain DOCKER_CLI_HINTS=false GIT_TERMINAL_PROMPT=0

  cd "$ROOT"
  previous=$(git rev-parse --short HEAD)
  log "Deploying '$REF' in $ROOT (currently at $(git log -1 --oneline))"

  refuse_if_dirty

  log "Fetching and checking out '$REF'"
  checkout_ref
  log "Checked out $(git log -1 --oneline)"

  install_env_file

  log "Pulling the telegram-bot-api image (best effort; rich messages need Bot API >= 10.1)"
  docker compose pull --quiet telegram-bot-api || echo "WARNING: pull failed, keeping the current image"

  log "Building the bot image"
  docker compose build --pull bot

  log "Starting the stack"
  docker compose up -d --remove-orphans

  log "Waiting up to ${health_timeout}s for the bot to start"
  if ! wait_for_bot "$health_timeout"; then
    {
      echo "ERROR: the bot container is not running or did not log 'is running' within ${health_timeout}s."
      echo "The previous commit was $previous and the previous .env is .env.bak (manual rollback:"
      echo "git checkout $previous && mv .env.bak .env && docker compose up -d --build)."
      echo "Note: the first start on a fresh server also builds the sandbox image, which takes longer."
    } >&2
    docker compose ps
    docker compose logs --tail 100 --no-log-prefix bot
    exit 1
  fi

  log "Bot is running"
  docker compose ps
  docker compose logs --tail 30 --no-log-prefix bot
  log "Deployed $(git log -1 --oneline)"

  docker image prune -f >/dev/null || true
}

main "$@" </dev/null

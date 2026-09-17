#!/usr/bin/env bash
# Reproduce the graders' submission test inside Docker.
#
# Section 8 of the brief describes two separate things: a Docker container for
# your own local testing, and pre-runs on the course's machines via MaMpf.
# This script is the first one. It mirrors what section 8 says the graders do
# with an uploaded .zip:
#
#   1. unzip it
#   2. install anything in requirements.txt        (we have none; numpy only)
#   3. find the first directory containing callbacks.py
#   4. copy that directory into their agent_code
#   5. run a single game with self.train = False against three random_agents
#
# Deliberately copies the agent in *from the zip* rather than relying on the
# `COPY . .` in the Dockerfile, so what gets tested is the artifact actually
# being uploaded, not the working tree it was built from.
#
# Usage:
#   bash scripts/docker_submission_test.sh [path/to/final-project-agent-code.zip]
#
# Requires Docker Desktop to be running. On Apple silicon, set
#   export DOCKER_PLATFORM="--platform linux/amd64"
# before running.

set -euo pipefail

ZIP="${1:-final-project-agent-code.zip}"
IMAGE="bomberman-submission-test"
CONTAINER="bomberman-submission-run"
ROUNDS="${ROUNDS:-5}"
PLATFORM="${DOCKER_PLATFORM:-}"

command -v docker >/dev/null || { echo "docker not on PATH"; exit 1; }
docker info >/dev/null 2>&1 || {
  echo "The Docker daemon is not reachable. Start Docker Desktop and retry."
  exit 1
}
[ -f "$ZIP" ] || { echo "no such zip: $ZIP"; exit 1; }

# The upstream Dockerfile currently fails: `FROM continuumio/miniconda3` is an
# unpinned tag that now resolves to a Python for which TensorFlow ships no
# wheels. Prefer the official file if it builds; fall back to the pinned local
# one. Once the course publishes a fix, `git pull` and the official file wins
# again automatically.
DOCKERFILE="Dockerfile"
echo "==> building with $DOCKERFILE (this takes a while the first time)"
if ! docker build $PLATFORM -f Dockerfile -t "$IMAGE" . ; then
  echo
  echo "==> official Dockerfile failed to build; retrying with Dockerfile.local"
  [ -f Dockerfile.local ] || { echo "Dockerfile.local not found"; exit 1; }
  DOCKERFILE="Dockerfile.local"
  docker build $PLATFORM -f Dockerfile.local -t "$IMAGE" .
fi

# Step 1/3: unzip and locate the first directory holding callbacks.py.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
unzip -q "$ZIP" -d "$WORK/unzipped"
AGENT_DIR="$(find "$WORK/unzipped" -name callbacks.py -printf '%h\n' | sort | head -1)"
[ -n "$AGENT_DIR" ] || { echo "no callbacks.py inside $ZIP"; exit 1; }
AGENT_NAME="$(basename "$AGENT_DIR")"
echo "==> agent found in zip: $AGENT_NAME"

if [ -f "$WORK/unzipped/requirements.txt" ]; then
  echo "==> zip ships a requirements.txt; the graders would install it here"
fi

# Step 5: one game, no --train, so callbacks.py sees self.train = False.
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker create $PLATFORM --name "$CONTAINER" "$IMAGE" \
  python main.py play --no-gui --n-rounds "$ROUNDS" \
  --agents "$AGENT_NAME" random_agent random_agent random_agent >/dev/null

# Step 4: overwrite whatever the image build baked in with the zip's copy.
docker cp "$AGENT_DIR" "$CONTAINER:/home/bomberman/agent_code/$AGENT_NAME"

echo "==> running $ROUNDS rounds vs three random_agents"
set +e
docker start -a "$CONTAINER"
RUN_STATUS=$?
set -e

# Section 8 steps 8-9: retrieve the logs.
docker cp "$CONTAINER:/home/bomberman/logs/game.log" ./docker_game.log 2>/dev/null || true
docker cp "$CONTAINER:/home/bomberman/agent_code/$AGENT_NAME/logs" ./docker_agent_logs 2>/dev/null || true
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo
echo "================ result ================"
echo "built with     : $DOCKERFILE"
echo "exit status    : $RUN_STATUS  (0 = the game ran to completion)"
if [ -f ./docker_game.log ]; then
  ERRORS=$(grep -ciE "traceback|error|exception" ./docker_game.log || true)
  SLOW=$(grep -ci "exceeded" ./docker_game.log || true)
  echo "game.log       : ./docker_game.log"
  echo "  errors       : $ERRORS   (expect 0)"
  echo "  slow steps   : $SLOW   (expect 0; the limit is 0.5 s per step)"
else
  echo "game.log       : not retrieved"
fi
[ -d ./docker_agent_logs ] && echo "agent logs     : ./docker_agent_logs/"
echo "========================================"
[ "$RUN_STATUS" -eq 0 ] || exit "$RUN_STATUS"

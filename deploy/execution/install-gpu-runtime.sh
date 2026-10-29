#!/usr/bin/env bash
# Register NVIDIA's OCI runtime with a live Docker reload; never restart Docker.
# Sources: NVIDIA Container Toolkit install guide; Docker daemon reload reference.
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo 'Run as root'; exit 1; }
runtime_version="${NVIDIA_CONTAINER_TOOLKIT_VERSION:-1.20.1-1}"
evidence_dir="${1:?Set a private, existing staging directory}"
mkdir -p "$evidence_dir/runtime-install"
chmod 700 "$evidence_dir/runtime-install"
docker ps -q | xargs -r docker inspect --format '{{.Id}} {{.State.StartedAt}}' > "$evidence_dir/runtime-install/containers-before.txt"
if [[ -f /etc/docker/daemon.json ]]; then
  cp -p /etc/docker/daemon.json "$evidence_dir/runtime-install/daemon-before.json"
fi
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l
if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey -o "$evidence_dir/runtime-install/repository-key"
  gpg --dearmor --yes -o /usr/share/keyrings/webkinpred-nvidia-container.gpg "$evidence_dir/runtime-install/repository-key"
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/webkinpred-nvidia-container.gpg] https://#g' \
    > /etc/apt/sources.list.d/webkinpred-nvidia-container.list
  apt-get update
  apt-get install -y --no-install-recommends --no-upgrade \
    "nvidia-container-toolkit=$runtime_version" \
    "nvidia-container-toolkit-base=$runtime_version" \
    "libnvidia-container-tools=$runtime_version" \
    "libnvidia-container1=$runtime_version"
fi
nvidia-ctk runtime configure --runtime=docker
dockerd --validate --config-file=/etc/docker/daemon.json
docker_pid="$(systemctl show docker -p MainPID --value)"
[[ "$docker_pid" =~ ^[1-9][0-9]*$ ]] || { echo 'Cannot identify Docker daemon'; exit 1; }
kill -HUP "$docker_pid"
for _ in {1..10}; do
  if docker info --format '{{json .Runtimes}}' | python3 -c 'import json,sys;sys.exit("nvidia" not in json.load(sys.stdin))'; then
    break
  fi
  sleep 1
done
docker info --format '{{json .Runtimes}}' | python3 -c 'import json,sys;assert "nvidia" in json.load(sys.stdin)'
while read -r container started; do
  [[ "$(docker inspect --format '{{.State.StartedAt}}' "$container")" == "$started" ]] || { echo "Container changed during installation: $container"; exit 1; }
done < "$evidence_dir/runtime-install/containers-before.txt"
nvidia-ctk --version > "$evidence_dir/runtime-install/version.txt"
echo 'NVIDIA runtime registered; existing containers retain their original start times.'

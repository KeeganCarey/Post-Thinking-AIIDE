#!/bin/bash

#download uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# start a virtual env and activate
uv python install 3.13
uv venv
source .venv/bin/activate

# sync the libraries
uv sync

# better performance
echo "enable transparent hugepages"
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"

echo ""
echo "Set secrets"
echo "export HF_TOKEN=xxx >> ~/.bashrc"
echo "source ~/.bashrc"

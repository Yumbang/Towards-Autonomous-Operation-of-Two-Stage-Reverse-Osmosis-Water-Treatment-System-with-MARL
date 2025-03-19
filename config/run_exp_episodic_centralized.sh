#!/bin/bash

# Directory containing configuration files
CONFIG_DIR="./episodic_conf"

# Python script to run. Centralized version also use same config file.
PYTHON_SCRIPT="/home/ybang-eai/research/2024/ROMARL/ROMARL/optimize_pressure_RO_centralized.py"

# Loop through all YAML files in the config directory
for config_file in "$CONFIG_DIR"/centralized_config_*.yaml; do
  # Check if the file exists to avoid errors if there are no matching files
  if [ -f "$config_file" ]; then
    echo "Running $PYTHON_SCRIPT with configuration $config_file"
    python "$PYTHON_SCRIPT" --yaml_file "$config_file" --description "Experimenting reward ratio."
  else
    echo "No configuration files found in $CONFIG_DIR"
  fi
done

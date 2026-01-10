#!/bin/bash
# Serve documentation locally for development

set -e

# Check if mkdocs is installed
if ! command -v mkdocs &> /dev/null; then
    echo "Installing documentation dependencies..."
    pip install -r docs/requirements.txt
fi

echo "Starting documentation server..."
echo "Open http://localhost:8000 in your browser"
echo "Press Ctrl+C to stop"

mkdocs serve --dev-addr 0.0.0.0:8000

#!/bin/bash
# Build documentation for production

set -e

# Check if mkdocs is installed
if ! command -v mkdocs &> /dev/null; then
    echo "Installing documentation dependencies..."
    pip install -r docs/requirements.txt
fi

echo "Building documentation..."
mkdocs build --strict

echo "Documentation built successfully!"
echo "Output: site/"

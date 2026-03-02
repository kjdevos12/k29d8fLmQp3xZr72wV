#!/bin/bash
# Get the path of the current folder
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Move files to Desktop
mv "$SCRIPT_DIR/esp.py" ~/Desktop/
mv "$SCRIPT_DIR/best.pt" ~/Desktop/

echo "Files moved to Desktop!"
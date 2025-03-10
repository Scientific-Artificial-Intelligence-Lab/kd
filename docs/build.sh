#!/bin/bash

# Remove old build
rm -rf build/

# Create necessary directories
mkdir -p source/_static
mkdir -p source/_templates

# Build documentation
make clean
make html

# Check build status
if [ $? -eq 0 ]; then
    echo "Documentation built successfully!"
    echo "Open build/html/index.html to view"
else
    echo "Build failed!"
fi

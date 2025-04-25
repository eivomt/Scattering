#!/bin/bash

# Frame rate (change as needed)
FPS=30

# Loop through directories ./png/1 to ./png/15
for i in {1..15}; do
    INPUT_DIR="./png/$i"
    OUTPUT_FILE="wave_packet_$i.mp4"

    if [ -d "$INPUT_DIR" ]; then
        echo "Processing directory: $INPUT_DIR"
        ffmpeg -framerate $FPS -i "$INPUT_DIR/%03d.png" -c:v libx264 -pix_fmt yuv420p -y "$OUTPUT_FILE"
    else
        echo "Directory $INPUT_DIR does not exist, skipping."
    fi
done
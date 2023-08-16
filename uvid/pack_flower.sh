#!/bin/bash
ffmpeg -i soccer_cif.y4m -f rawvideo -pixel_format yuv420p - > input_video.raw
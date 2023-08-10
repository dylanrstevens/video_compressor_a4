#!/bin/bash
rm output_video.y4m;
ffmpeg -f rawvideo -pixel_format yuv420p -framerate 30 -video_size 352x288 -i - -f yuv4mpegpipe output_video.y4m < output_video.raw
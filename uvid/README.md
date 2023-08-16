
# CSC485B - Assignment 4 - video compressor and decompressor                          
# Dylan Stevens                                                                       
# V00957595                                                                           


## Overview

Video compressor and decompressor that supports I frames, P frames, motion compensation, run length encoding, delta encoding, and dynamic huffman coding. Various data structures and algorithms are used, including binary trees for huffman encoding, matrix multiplication in the discrete cosine transform, and many others.


## Features Implemented

**Basic Requirements**

1. Temporally compressed frames (P-frames)
- In the compressor:
    - Every 4 frames after an I frame, are encoded as P frames (See lines 1367 - 1560)

    - Previous frame is decompressed in the compressor (lines 1374 - 1376) and (lines 238 - 447 for decompression function in compressor)

    - DCT is performed on the delta values (lines 1548 - 1550)

- In the decompressor:
    - Every 4 frames after an I frame, are encoded as P frames (See lines 740 - 960)

    - The values are added back onto the last frame to create the current frame (lines 883, 905, 927)


2. Motion compensation 
- In the compressor:
    - Creates different motion vectors for every 16x16 block (lines 1394 - 1396) and (lines 1220 - 1265 for the function creating the motion vectors by minimum sum of absolute differences algorithm) 

    - Motion vector difference is added onto pixel for last frame (lines 1508, 1525, 1541)

- In the decompressor:
    - Motion vectors are added back onto last frame once recovered (lines 883, 905, 927)

3. Compression ratio of at least 12 achieved (with medium quality setting)
    - news_cif.y4m is compressed from 45621044 bytes to 2926137
    - soccer_cif.y4m is compressed from 45621038 bytes to 3661160
    - Many other examples


**Advanced requirements**

4. Dynamic Huffman Coding
- In the compressor:
    - Bit lengths are calculated based on frequencies (lines 140 - 222 for bit length function) and (lines 460 - 594 for prefix code output function)

- In the decompressor:
    - Once bitlengths are recovered, symbols are retrieved using the appropriate prefix code (lines 142 - 189)

5. Implementation achieves real time decompression for video samples of resolution 640 x 480 at 30 frames per second


## Architecture

The pipeline for the compressor is as follows:

- The very first frame is encoded as an I frame
- The next 4 frames after it are encoded as P frames, compressed as the difference of each one before it
- Another I frame is then encoded, followed by 4 more P frames, and the cycle repeats

For each frame (both I and P frames):

- Each color plane is divided into 16x16 macroblocks
- Bitlengths for every symbol in the frame are calculated based on frequencies, and dynamic huffman codes are created for every frame.
- Inside each block:
	- The discrete cosine transform is performed, with different 16x16 quantization matrices for each quality setting (low, medium, high)
	- The first two values are encoded as u16's
	- Every value after is encoded as a delta of the one before it
	- Every delta value then has run length encoding specifically for runs of zeros applied.
	- Every value and run length is then dynamically huffman encoded using the huffman codes for that particular frame
	- Bitstream for colour plane is pushed

For each P frame:

- Motion vectors are calculated for every block (before the DCT), using a 16 pixel radius around every 16x16 block
- The motion vectors are found using the minimum sum of absolute differences, in order to find the smallest delta values which will provide best compression in the quanitzation phase


## Bitstream

- video height: 32 bytes
- video width: 32 bytes
- quality level: 1 byte

For every I frame:
- frame indicator flag: 1 byte
- For every color plane:
	- lowest symbol in frame: 16 bytes (for sending dynamic codes)
	- highest symbol in frame: 16 bytes (for sending dynamic codes)
	- bit lengths for dynamic huffman codes: 5 bytes for each bitlength from lowest to highest
	- run length symbol: 5 bytes
	- number of bits in final compressed colour plane: 32 bytes
	- bitstream for colour plane: however many bits previously mentioned (run lengths are encoded as the symbol, and then a byte representing the run length for up to a maximum of 255 0's in a row. If more than 255 0's appear, another run length symbol and byte are encoded, and so on. symbols are encoded as their dynamic huffman code).
	
For every P frame:
Same every thing is the same as I frame, except there is a motion vector table encoded after the indicator flag, and before every color plane

The vector table is encoded in the bitstream as follows:

- Motion vectors for every block in Y-plane: bytes equal to number of blocks in a frame * 10 bytes per motion vector (5 for dx and 5 for dy)

- Motion vectors for every block in Cb-plane: bytes equal to number of blocks in a frame / 2 * 10 bytes per motion vector (5 for dx and 5 for dy)

- Motion vectors for every block in Cr-plane: bytes equal to number of blocks in a frame / 2 * 10 bytes per motion vector (5 for dx and 5 for dy)


## Bibliography

Lecture videos and slides provided by Bill Bird

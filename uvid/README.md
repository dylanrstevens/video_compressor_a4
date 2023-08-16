The pipeline for the compressor is as follows:

- The very first frame is encoded as an I frame
- The next 4 frames after it are encoded as P frames, compressed as the difference of each one before it
- Another I frame is then encoded and the cycle repeats

For each frame (both I and P frames):

- Each color plane is divided into 16x16 macroblocks
- Bitlengths for every symbol in the frame are calculated based on frequencies, and dynamic huffman codes are created for every frame.
- Inside each block:
	- The discrete cosine transform is performed, with different 16x16 quantization 	matrices for each quality setting (low, medium, high)
	- The first two values are encoded as u16's
	- Every value after is encoded as a delta of the one before it
	- Every delta value then has run length encoding specifically for runs of zeros 	applied.
	- Every value and run length is then dynamically huffman encoded using the 	huffman codes for that particular frame
	- Bitstream for colour plane is pushed

For each P frame:

- Motion vectors are calculated for every block (before the DCT), using a 16 pixel radius around every 16x16 block
- The motion vectors are found using the minimum sum of absolute differences, in order to find the smallest delta values which will provide best compression in the quanitzation phase

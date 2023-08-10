/* uvid_decompress.cpp
   CSC 485B/578B - Data Compression - Summer 2023

   Starter code for Assignment 4
   
   This placeholder code reads the (basically uncompressed) data produced by
   the uvid_compress starter code and outputs it in the uncompressed 
   YCbCr (YUV) format used for the sample video input files. To play the 
   the decompressed data stream directly, you can pipe the output of this
   program to the ffplay program, with a command like 

     ffplay -f rawvideo -pixel_format yuv420p -framerate 30 -video_size 352x288 - 2>/dev/null
   (where the resolution is explicitly given as an argument to ffplay).

   B. Bird - 2023-07-08
*/

#include <iostream>
#include <fstream>
#include <bitset>
#include <cmath>
#include <cstring>
#include <array>
#include <map>
#include <string>
#include <cassert>
#include <cstdint>
#include <tuple>
#include "input_stream.hpp"
#include "yuv_stream.hpp"

template<typename T>
std::vector<std::vector<T> > create_2d_vector(unsigned int outer, unsigned int inner){
    std::vector<std::vector<T> > V {outer, std::vector<T>(inner,T() )};
    return V;
}

std::map<int, unsigned int> generateStaticCodeLengths(int maxSymbol) {
    std::map<int, unsigned int> staticCodeLengths;

    // Symbol 0 gets 1 bit
    staticCodeLengths[0] = 1;

    // Symbols -1 and 1 get 3 bits
    staticCodeLengths[-1] = staticCodeLengths[1] = 3;

    // Symbols 2, 3, -2, and -3 get 5 bits
    staticCodeLengths[2] = staticCodeLengths[3] = staticCodeLengths[-2] = staticCodeLengths[-3] = 5;

    // For every other symbol, assign 11 bits
    for (int i = -maxSymbol; i <= maxSymbol; i++) {
        if (i != 0 && staticCodeLengths[i] == 0) {
            staticCodeLengths[i] = 13;
        }
    }

    return staticCodeLengths;
}

void printCodeLengths(const std::map<int, unsigned int>& codeLengths) {
    std::clog << "Symbol\tCode Length" << std::endl;
    for (const auto& entry : codeLengths) {
        int symbol = entry.first;
        unsigned int codeLength = entry.second;
        std::clog << symbol << "\t" << codeLength << std::endl;
    }
}

void printHuffmanTable(const std::map<int, unsigned int>& huffmanTable) {
    std::clog << "Symbol\tCode" << std::endl;
    for (const auto& entry : huffmanTable) {
        int symbol = entry.first;
        unsigned int code = entry.second;

        // Determine the number of bits required for the code
        int numBits = 1;
        unsigned int tempCode = code;
        while (tempCode >>= 1) {
            numBits++;
        }

        std::clog << symbol << "\t" << std::bitset<32>(code).to_string().substr(32 - numBits) << std::endl;
    }
}

std::map<int, unsigned int> createHuffmanTable(const std::map<int, unsigned int>& bitLengths) {
    std::map<int, unsigned int> huffmanTable;

    // Create code values based on bit lengths in ascending order
    std::vector<std::pair<unsigned int, int>> sortedLengths;
    for (const auto& entry : bitLengths) {
        unsigned int length = entry.second;
        int symbol = entry.first;
        sortedLengths.push_back(std::make_pair(length, symbol));
    }
    std::sort(sortedLengths.begin(), sortedLengths.end());

    unsigned int codeValue = 0;
    unsigned int maxLength = sortedLengths.back().first;
    for (unsigned int length = 1; length <= maxLength; length++) {
        unsigned int maxCodeValue = (1 << length) - 1;
        for (const auto& pair : sortedLengths) {
            if (pair.first == length) {
                if (codeValue > maxCodeValue) {
                    // Adjust codeValue to stay within the limit for this bit length
                    codeValue = maxCodeValue;
                }
                huffmanTable[pair.second] = codeValue;
                codeValue++;
            }
        }
        codeValue <<= 1;
    }

    return huffmanTable;
}


std::vector<int> decodeData(const std::vector<unsigned int>& bitstream, const std::map<int, unsigned int>& huffmanTable, unsigned int run_len_symbol, unsigned int run_len_symbol_len) {
    std::vector<int> decodedData;
    unsigned int numBits = 0;
    unsigned int currentCode = 0;

    std::map<unsigned int, int> reverseTable;
    for (const auto& entry : huffmanTable) {
        int symbol = entry.first;
        unsigned int code = entry.second;
        reverseTable[code] = symbol;
    }
    int run_found = 0;

    for (unsigned int bit : bitstream) {
        currentCode = (currentCode << 1) | bit;
        numBits++;

        if (!run_found) {
            if (currentCode == run_len_symbol) {
                run_found = 1;
                currentCode = 0;
                numBits = 0;
            }
            else {
                auto it = reverseTable.find(currentCode);
                if (it != reverseTable.end()) {
                    //std::clog << "code found: " << currentCode << "\n";
                    decodedData.push_back(it->second);
                    numBits = 0;
                    currentCode = 0;
                }
            }
        }
        else {
            if (numBits == 8) {
                //std::clog << "Run found: " << currentCode << "\n";
                for (unsigned int j = 0; j < currentCode; j++) {
                    decodedData.push_back(0);
                }
                numBits = 0;
                currentCode = 0;
                run_found = 0;
            }
        }
    }

    return decodedData;
}

void Delta_reverse_DCT(std::vector<std::vector<int>>& color_plane, InputBitStream stream, unsigned int q_id) {
    unsigned int height = color_plane.size();
    unsigned int width = color_plane[0].size();

    int m = 8;
    int n = 8;

    // Calculate the number of blocks in the image
    unsigned int num_blocks_x = (width + m - 1) / m;   // Round up the division
    unsigned int num_blocks_y = (height + n - 1) / n;  // Round up the division

    // Define the size of the DCT matrix
    int C_size = 8;

    // Define and calculate the DCT matrix C
    float C[C_size][C_size];
    float inv_sqrt_n = 1.0f / sqrt(C_size);
    float sqrt_2_over_n = sqrt(2.0f) / sqrt(C_size);
    for (int i = 0; i < C_size; i++) {
        for (int j = 0; j < C_size; j++) {
            if (i == 0)
                C[i][j] = inv_sqrt_n;
            else
                C[i][j] = sqrt_2_over_n * cos((2 * j + 1) * i * M_PI / (2 * C_size));
        }
    }

    // Calculate the transpose of C
    float C_transpose[C_size][C_size];
    for (int i = 0; i < C_size; i++) {
        for (int j = 0; j < C_size; j++) {
            C_transpose[j][i] = C[i][j];
        }
    }

    // Create the quantization matrix (Q) for the block
    int Q_med[8][8] = {
        {32, 20, 20, 32, 48, 80, 100, 120},
        {24, 24, 28, 36, 52, 116, 120, 108},
        {28, 24, 32, 48, 80, 112, 136, 112},
        {28, 32, 44, 56, 100, 172, 160, 124},
        {36, 44, 72, 112, 136, 216, 204, 152},
        {48, 68, 108, 128, 160, 208, 224, 184},
        {96, 128, 156, 172, 204, 240, 240, 200},
        {144, 184, 188, 196, 224, 200, 204, 196}};

    int Q_hi[8][8] = {
        {8, 5, 5, 8, 12, 20, 25, 30},
        {6, 6, 7, 9, 13, 29, 30, 27},
        {7, 6, 8, 12, 20, 28, 34, 28},
        {7, 8, 11, 14, 25, 43, 40, 31},
        {9, 11, 18, 28, 34, 54, 51, 38},
        {12, 17, 27, 32, 40, 52, 56, 46},
        {24, 32, 39, 43, 51, 60, 60, 50},
        {36, 46, 47, 49, 56, 50, 51, 49}};

    int Q_low[8][8] = {
    {64, 40, 40, 64, 96, 160, 200, 240},
    {48, 48, 56, 72, 104, 232, 240, 216},
    {56, 48, 64, 96, 160, 224, 255, 224},
    {56, 64, 88, 112, 200, 255, 255, 248},
    {72, 88, 144, 224, 255, 255, 255, 255},
    {96, 136, 216, 255, 255, 255, 255, 255},
    {192, 255, 255, 255, 255, 255, 255, 255},
    {255, 255, 255, 255, 255, 255, 255, 255}
};

    std::vector<unsigned int> bits;

    //retrieve the dynamic codes
    std::map<int, unsigned int> bit_lengths_for_deltas;

    int lowest = static_cast<int16_t>(stream.read_u16());
    int highest = static_cast<int16_t>(stream.read_u16());

    //std::clog << "p frame\n";

    //std::clog << lowest << "\n";
    //std::clog << highest << "\n";

    
    for (int sym_iter = lowest; sym_iter <= highest; sym_iter++) {
        unsigned int currentCode = 0;
        for (unsigned int j = 0; j < 5; j++) {
            unsigned int v = stream.read_bit();
            currentCode = (currentCode << 1) | v;
        }
        if (currentCode != 0) {
            bit_lengths_for_deltas[sym_iter] = currentCode;
        }
    }
    unsigned int currentCode = 0;
    for (unsigned int j = 0; j < 5; j++) {
        unsigned int v = stream.read_bit();
        currentCode = (currentCode << 1) | v;
    }

    unsigned int run_len_symbol_len = currentCode;
    bit_lengths_for_deltas[513] = currentCode;

    u32 num_of_bits_in_frame_plane = stream.read_u32();

    //std::clog << num_of_bits_in_frame_plane << "\n";
    
    for (unsigned int i = 0; i < num_of_bits_in_frame_plane; i++) {
        unsigned int bit = stream.read_bit();
        bits.push_back(bit);
    }
    //std::map<int, unsigned int> bit_lengths_for_deltas = generateStaticCodeLengths(512);
    std::map<int, unsigned int> huffmanTable = createHuffmanTable(bit_lengths_for_deltas);

    unsigned int run_len_symbol = huffmanTable[513];
    huffmanTable.erase(513);

    //printHuffmanTable(huffmanTable);

    //std::clog<<"run len sym: "<<run_len_symbol<<", run len sym len: "<<run_len_symbol_len<<"\n";
    std::vector<int> decompressed_data = decodeData(bits, huffmanTable, run_len_symbol, run_len_symbol_len);

    /*
    std::clog << "decompressed: \n";
    for (auto d : decompressed_data) {
        std::clog << d << "    ";
    }
    std::clog << std::endl;
    */


    int data_iterator = 0;

    // Iterate over each block in the image
    for (unsigned int block_y = 0; block_y < num_blocks_y; block_y++) {
        for (unsigned int block_x = 0; block_x < num_blocks_x; block_x++) {

            //u16 dct_const = stream.read_u16();
            //u16 first_val = stream.read_u16();

            //std::clog << dct_const << "\n";

            int last_val = 0;
            int original_val = 0;
            int quantized_dct[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if ((i == 0) && (j == 0 || j == 1)) {
                        u16 val = stream.read_u16();
                        original_val = val;
                        quantized_dct[i][j] = static_cast<int16_t>(val);
                        last_val = static_cast<int16_t>(val);
                    } else {
                        int val = decompressed_data[data_iterator];
                        val = val + last_val;
                        original_val = val;
                        quantized_dct[i][j] = val;
                        last_val = original_val;
                        data_iterator++;
                    }
                }
                
            }

            /*
            if (block_y == num_blocks_y - 1) {
                std::clog << "Block ("<<block_y+1<<", "<<block_x+1<<") values for frame: \n";
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        std::clog << std::round(quantized_dct[i][j]) << "    ";
                    }
                    std::clog << std::endl;
                }
                std::clog << std::endl;
            }
            */
            

             /*
            std::clog << "quantized block:\n";
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    std::clog << quantized_dct[i][j] << "    ";
                }
                std::clog << std::endl;
            }
            std::clog << std::endl;
          
           
            
            // Get quanitzed matrix from input
            int quantized_dct[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (i == 0 && j == 0) {
                        u16 val = stream.read_u16();
                        quantized_dct[i][j] = static_cast<int16_t>(val);
                    } else {
                        unsigned int prefix_bit = stream.read_bit();
                        if (prefix_bit == 1) {
                            u16 val = stream.read_u16();
                            quantized_dct[i][j] = static_cast<int16_t>(val);
                        } else if (prefix_bit == 0) {
                            quantized_dct[i][j] = 0;
                        }
                    }
                }
            }
            */
            

            int unquantized_dct[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (q_id == 0) {
                        unquantized_dct[i][j] = quantized_dct[i][j] * Q_low[i][j];
                    } else if (q_id == 1) {
                        unquantized_dct[i][j] = quantized_dct[i][j] * Q_med[i][j];
                    } else if (q_id == 2) {
                        unquantized_dct[i][j] = quantized_dct[i][j] * Q_hi[i][j];
                    }
                }
            }

            // Compute the matrix multiplication of (C^T)(block)
            float temp[C_size][C_size];
            memset(temp, 0, sizeof(temp));  // Initialize all elements of temp to zero
            for (int i = 0; i < C_size; i++) {
                for (int j = 0; j < C_size; j++) {
                    for (int k = 0; k < C_size; k++) {
                        temp[i][j] += C_transpose[i][k] * unquantized_dct[k][j];
                    }
                }
            }

            // Compute the matrix multiplication of (C^T)(block)(C)
            float dct_block[C_size][C_size];
            memset(dct_block, 0, sizeof(dct_block));  // Initialize all elements of dct_block to zero
            for (int i = 0; i < C_size; i++) {
                for (int j = 0; j < C_size; j++) {
                    for (int k = 0; k < C_size; k++) {
                        dct_block[i][j] += temp[i][k] * C[k][j];
                    }
                }
            }

            /*
            std::clog << "P frame deltas for block ("<<block_y+1<<", "<<block_x+1<<")\n";
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    std::clog << std::round(dct_block[i][j]) << "    ";
                }
                std::clog << std::endl;
            }
            std::clog << std::endl;
            */
            

            // Update the color_plane with the inverse DCT values
            for (int i = 0; i < m && (block_y * 8 + i) < height; i++) {
                for (int j = 0; j < n && (block_x * 8 + j) < width; j++) {
                    int rounded = std::round(dct_block[i][j]);
                    color_plane.at(block_y * 8 + i).at(block_x * 8 + j) = rounded;
                }
            }
            
        }
    }
}


void reverse_DCT(std::vector<std::vector<unsigned char>>& color_plane, InputBitStream stream, unsigned int q_id) {
    unsigned int height = color_plane.size();
    unsigned int width = color_plane[0].size();

    int m = 8;
    int n = 8;

    // Calculate the number of blocks in the image
    unsigned int num_blocks_x = (width + m - 1) / m;   // Round up the division
    unsigned int num_blocks_y = (height + n - 1) / n;  // Round up the division

    // Define the size of the DCT matrix
    int C_size = 8;

    // Define and calculate the DCT matrix C
    float C[C_size][C_size];
    float inv_sqrt_n = 1.0f / sqrt(C_size);
    float sqrt_2_over_n = sqrt(2.0f) / sqrt(C_size);
    for (int i = 0; i < C_size; i++) {
        for (int j = 0; j < C_size; j++) {
            if (i == 0)
                C[i][j] = inv_sqrt_n;
            else
                C[i][j] = sqrt_2_over_n * cos((2 * j + 1) * i * M_PI / (2 * C_size));
        }
    }

    // Calculate the transpose of C
    float C_transpose[C_size][C_size];
    for (int i = 0; i < C_size; i++) {
        for (int j = 0; j < C_size; j++) {
            C_transpose[j][i] = C[i][j];
        }
    }

    // Create the quantization matrix (Q) for the block
    int Q_med[8][8] = {
        {32, 20, 20, 32, 48, 80, 100, 120},
        {24, 24, 28, 36, 52, 116, 120, 108},
        {28, 24, 32, 48, 80, 112, 136, 112},
        {28, 32, 44, 56, 100, 172, 160, 124},
        {36, 44, 72, 112, 136, 216, 204, 152},
        {48, 68, 108, 128, 160, 208, 224, 184},
        {96, 128, 156, 172, 204, 240, 240, 200},
        {144, 184, 188, 196, 224, 200, 204, 196}};

    int Q_hi[8][8] = {
        {8, 5, 5, 8, 12, 20, 25, 30},
        {6, 6, 7, 9, 13, 29, 30, 27},
        {7, 6, 8, 12, 20, 28, 34, 28},
        {7, 8, 11, 14, 25, 43, 40, 31},
        {9, 11, 18, 28, 34, 54, 51, 38},
        {12, 17, 27, 32, 40, 52, 56, 46},
        {24, 32, 39, 43, 51, 60, 60, 50},
        {36, 46, 47, 49, 56, 50, 51, 49}};

    int Q_low[8][8] = {
    {64, 40, 40, 64, 96, 160, 200, 240},
    {48, 48, 56, 72, 104, 232, 240, 216},
    {56, 48, 64, 96, 160, 224, 255, 224},
    {56, 64, 88, 112, 200, 255, 255, 248},
    {72, 88, 144, 224, 255, 255, 255, 255},
    {96, 136, 216, 255, 255, 255, 255, 255},
    {192, 255, 255, 255, 255, 255, 255, 255},
    {255, 255, 255, 255, 255, 255, 255, 255}
};

    std::vector<unsigned int> bits;

    std::map<int, unsigned int> bit_lengths_for_deltas;

    int lowest = static_cast<int16_t>(stream.read_u16());
    int highest = static_cast<int16_t>(stream.read_u16());

    //std::clog << "I frame\n";

    //std::clog << lowest << "\n";
    //std::clog << highest << "\n";

    
    for (int sym_iter = lowest; sym_iter <= highest; sym_iter++) {
        unsigned int currentCode = 0;
        for (unsigned int j = 0; j < 5; j++) {
            unsigned int v = stream.read_bit();
            currentCode = (currentCode << 1) | v;
        }
        if (currentCode != 0) {
            
            bit_lengths_for_deltas[sym_iter] = currentCode;
        }
    }
    unsigned int currentCode = 0;
    for (unsigned int j = 0; j < 5; j++) {
        unsigned int v = stream.read_bit();
        currentCode = (currentCode << 1) | v;
    }

    unsigned int run_len_symbol_len = currentCode;
    bit_lengths_for_deltas[513] = currentCode;

    u32 num_of_bits_in_frame_plane = stream.read_u32();

    //std::clog << num_of_bits_in_frame_plane << "\n";
    
    for (unsigned int i = 0; i < num_of_bits_in_frame_plane; i++) {
        unsigned int bit = stream.read_bit();
        bits.push_back(bit);
    }
    //std::map<int, unsigned int> bit_lengths_for_deltas = generateStaticCodeLengths(512);
    std::map<int, unsigned int> huffmanTable = createHuffmanTable(bit_lengths_for_deltas);

    //printHuffmanTable(huffmanTable);
    //printCodeLengths(bit_lengths_for_deltas);

    unsigned int run_len_symbol = huffmanTable[513];
    huffmanTable.erase(513);

    
    //std::clog<<"run len sym: "<<run_len_symbol<<", run len sym len: "<<run_len_symbol_len<<"\n";
    std::vector<int> decompressed_data = decodeData(bits, huffmanTable, run_len_symbol, run_len_symbol_len);

    /*
    std::clog << "decompressed: \n";
    for (auto d : decompressed_data) {
        std::clog << d << "    ";
    }
    std::clog << std::endl;
    */
    

    int data_iterator = 0;

    // Iterate over each block in the image
    for (unsigned int block_y = 0; block_y < num_blocks_y; block_y++) {
        for (unsigned int block_x = 0; block_x < num_blocks_x; block_x++) {

            //u16 dct_const = stream.read_u16();
            //u16 first_val = stream.read_u16();

            //std::clog << dct_const << "\n";

            int last_val = 0;
            int original_val = 0;
            int quantized_dct[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if ((i == 0) && (j == 0 || j == 1)) {
                        u16 val = stream.read_u16();
                        original_val = val;
                        quantized_dct[i][j] = static_cast<int16_t>(val);
                        last_val = static_cast<int16_t>(val);
                    } else {
                        int val = decompressed_data[data_iterator];
                        val = val + last_val;
                        original_val = val;
                        quantized_dct[i][j] = val;
                        last_val = original_val;
                        data_iterator++;
                    }
                }
                
            }

            /*
            std::clog << "quantized block:\n";
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    std::clog << quantized_dct[i][j] << "    ";
                }
                std::clog << std::endl;
            }
            std::clog << std::endl;
            */
           
           /* 
            // Get quanitzed matrix from input
            int quantized_dct[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (i == 0 && j == 0) {
                        u16 val = stream.read_u16();
                        quantized_dct[i][j] = static_cast<int16_t>(val);
                    } else {
                        unsigned int prefix_bit = stream.read_bit();
                        if (prefix_bit == 1) {
                            u16 val = stream.read_u16();
                            quantized_dct[i][j] = static_cast<int16_t>(val);
                        } else if (prefix_bit == 0) {
                            quantized_dct[i][j] = 0;
                        }
                    }
                }
            }
            */
            

            int unquantized_dct[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (q_id == 0) {
                        unquantized_dct[i][j] = quantized_dct[i][j] * Q_low[i][j];
                    } else if (q_id == 1) {
                        unquantized_dct[i][j] = quantized_dct[i][j] * Q_med[i][j];
                    } else if (q_id == 2) {
                        unquantized_dct[i][j] = quantized_dct[i][j] * Q_hi[i][j];
                    }
                }
            }

            // Compute the matrix multiplication of (C^T)(block)
            float temp[C_size][C_size];
            memset(temp, 0, sizeof(temp));  // Initialize all elements of temp to zero
            for (int i = 0; i < C_size; i++) {
                for (int j = 0; j < C_size; j++) {
                    for (int k = 0; k < C_size; k++) {
                        temp[i][j] += C_transpose[i][k] * unquantized_dct[k][j];
                    }
                }
            }

            // Compute the matrix multiplication of (C^T)(block)(C)
            float dct_block[C_size][C_size];
            memset(dct_block, 0, sizeof(dct_block));  // Initialize all elements of dct_block to zero
            for (int i = 0; i < C_size; i++) {
                for (int j = 0; j < C_size; j++) {
                    for (int k = 0; k < C_size; k++) {
                        dct_block[i][j] += temp[i][k] * C[k][j];
                    }
                }
            }

            // Update the color_plane with the inverse DCT values
            for (int i = 0; i < m && (block_y * 8 + i) < height; i++) {
                for (int j = 0; j < n && (block_x * 8 + j) < width; j++) {
                    int rounded = std::round(dct_block[i][j]);
                    if (rounded < 0) {
                        color_plane.at(block_y * 8 + i).at(block_x * 8 + j) = 0;
                    } else if (rounded > 255) {
                        color_plane.at(block_y * 8 + i).at(block_x * 8 + j) = 255;
                    } else {
                        color_plane.at(block_y * 8 + i).at(block_x * 8 + j) = rounded;
                    }
                }
            }
            
        }
    }
}


struct Frame_track {
    unsigned int type;
    // 0 for I frame, 1 for P frame
    std::vector<std::vector<unsigned char>> Y;
    std::vector<std::vector<unsigned char>> Cb;
    std::vector<std::vector<unsigned char>> Cr;

    Frame_track(unsigned int type, std::vector<std::vector<unsigned char>> Y, std::vector<std::vector<unsigned char>> Cb, std::vector<std::vector<unsigned char>> Cr) {
        this->type = type;
        this->Y = Y;
        this->Cb = Cb;
        this->Cr = Cr;
    }
};


void print2DArray(const std::vector<std::vector<unsigned char>>& array) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            std::clog << static_cast<unsigned int>(array.at(i).at(j)) << "    ";
        }
        std::clog << std::endl;
    }
    std::clog << std::endl;
}

void print2DdeltaArray(const std::vector<std::vector<int>>& array) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            std::clog << (array.at(i).at(j)) << "    ";
        }
        std::clog << std::endl;
    }
    std::clog << std::endl;
}


struct MotionVector {
    int dx; // Horizontal motion vector
    int dy; // Vertical motion vector
};



int main(int argc, char** argv){

    //Note: This program must not take any command line arguments. (Anything
    //      it needs to know about the data must be encoded into the bitstream)
    
    InputBitStream input_stream {std::cin};


    u32 height {input_stream.read_u32()};
    u32 width {input_stream.read_u32()};

    unsigned int q_id = input_stream.read_byte();


    YUVStreamWriter writer {std::cout, width, height};

    Frame_track last_frame(1, {{0}}, {{0}}, {{0}});

    int frame_type = 4;

    int frame_count = 0;

    while (input_stream.read_byte()){

        YUVFrame420& frame = writer.frame();

        //std::clog << "frame: "<<frame_count<<"\n";

        if (frame_type == 4) {

            auto Y = create_2d_vector<unsigned char>(height, width);
            auto Cb = create_2d_vector<unsigned char>(height/2, width/2);
            auto Cr = create_2d_vector<unsigned char>(height/2, width/2);

            reverse_DCT(Y, input_stream, q_id);
            reverse_DCT(Cb, input_stream, q_id);
            reverse_DCT(Cr, input_stream, q_id);

            last_frame.type = 0;
            last_frame.Y = Y;
            last_frame.Cb = Cb;
            last_frame.Cr = Cr;

            for (u32 y = 0; y < height; y++)
                for (u32 x = 0; x < width; x++)
                    frame.Y(x,y) = Y.at(y).at(x);
            for (u32 y = 0; y < height/2; y++)
                for (u32 x = 0; x < width/2; x++)
                    frame.Cb(x,y) = Cb.at(y).at(x);
            for (u32 y = 0; y < height/2; y++)
                for (u32 x = 0; x < width/2; x++)
                    frame.Cr(x,y) = Cr.at(y).at(x);
            writer.write_frame();

            frame_type = 0;
        }
        else if (frame_type < 4) {


            auto last_y = last_frame.Y;
            auto last_cb = last_frame.Cb;
            auto last_cr = last_frame.Cr;

            auto Y = create_2d_vector<unsigned char>(height, width);
            auto Cb = create_2d_vector<unsigned char>(height/2, width/2);
            auto Cr = create_2d_vector<unsigned char>(height/2, width/2);

            auto un_delta_Y = create_2d_vector<int>(height, width);
            auto un_delta_Cb = create_2d_vector<int>(height/2, width/2);
            auto un_delta_Cr = create_2d_vector<int>(height/2, width/2);

            
            std::vector<MotionVector> motion_vectors_y;
            std::vector<MotionVector> motion_vectors_cb;
            std::vector<MotionVector> motion_vectors_cr;


            for (u32 i = 0; i < (height*width/8/8); i++) {
                
                int sign_num;
                unsigned int sign_bit = input_stream.read_bit();
                if (sign_bit == 1) {
                    sign_num = 1;
                }
                else {
                    sign_num = -1;
                }
                //read in 3 bits
                unsigned int dx = 0;
                for (int j = 3; j >= 0; j--) {
                    unsigned int bit = input_stream.read_bit();
                    dx = (dx << 1) | bit;
                }
                dx = dx*sign_num;

                sign_bit = input_stream.read_bit();
                if (sign_bit == 1) {
                    sign_num = 1;
                }
                else {
                    sign_num = -1;
                }
                //read in 3 bits
                unsigned int dy = 0;
                for (int j = 3; j >= 0; j--) {
                    unsigned int bit = input_stream.read_bit();
                    dy = (dy << 1) | bit;
                }
                dy = dy*sign_num;
                MotionVector vec(dx, dy);
                motion_vectors_y.push_back(vec);
            }

            //push cb motion vectors
            for (u32 i = 0; i < ((height/2)*(width/2)/8/8); i++) {

                int sign_num;
                unsigned int sign_bit = input_stream.read_bit();
                if (sign_bit == 1) {
                    sign_num = 1;
                }
                else {
                    sign_num = -1;
                }
                //read in 3 bits
                unsigned int dx = 0;
                for (int j = 3; j >= 0; j--) {
                    unsigned int bit = input_stream.read_bit();
                    dx = (dx << 1) | bit;
                }
                dx = dx*sign_num;

                sign_bit = input_stream.read_bit();
                if (sign_bit == 1) {
                    sign_num = 1;
                }
                else {
                    sign_num = -1;
                }
                //read in 3 bits
                unsigned int dy = 0;
                for (int j = 3; j >= 0; j--) {
                    unsigned int bit = input_stream.read_bit();
                    dy = (dy << 1) | bit;
                }
                dy = dy*sign_num;
                MotionVector vec(dx, dy);
                motion_vectors_cb.push_back(vec);            }

            //push cr motion vectors
            for (u32 i = 0; i < ((height/2)*(width/2)/8/8); i++) {

                int sign_num;
                unsigned int sign_bit = input_stream.read_bit();
                if (sign_bit == 1) {
                    sign_num = 1;
                }
                else {
                    sign_num = -1;
                }
                //read in 3 bits
                unsigned int dx = 0;
                for (int j = 3; j >= 0; j--) {
                    unsigned int bit = input_stream.read_bit();
                    dx = (dx << 1) | bit;
                }
                dx = dx*sign_num;

                sign_bit = input_stream.read_bit();
                if (sign_bit == 1) {
                    sign_num = 1;
                }
                else {
                    sign_num = -1;
                }
                //read in 3 bits
                unsigned int dy = 0;
                for (int j = 3; j >= 0; j--) {
                    unsigned int bit = input_stream.read_bit();
                    dy = (dy << 1) | bit;
                }
                dy = dy*sign_num;
                MotionVector vec(dx, dy);
                motion_vectors_cr.push_back(vec);
            }


            Delta_reverse_DCT(un_delta_Y, input_stream, q_id);
            Delta_reverse_DCT(un_delta_Cb, input_stream, q_id);
            Delta_reverse_DCT(un_delta_Cr, input_stream, q_id);

            //std::clog << "P Frame unconstructed\n";
            //print2DdeltaArray(un_delta_Y); 


            unsigned int blockIndex = 0;
            for (u32 block_y = 0; block_y < height; block_y += 8) {
                for (u32 block_x = 0; block_x < width; block_x += 8) {
                    //block level
                    int dx = motion_vectors_y[blockIndex].dx;
                    int dy = motion_vectors_y[blockIndex].dy;
                    for (u32 y = block_y; y < block_y + 8 && y < height; y++) {
                        for (u32 x = block_x; x < block_x + 8 && x < width; x++) {
                            int val = un_delta_Y.at(y).at(x) + last_y.at(y+dy).at(x+dx);
                            if (val > 255) {
                                val = 255;
                            }
                            else if (val < 0) {
                                val = 0;
                            }
                            Y.at(y).at(x) = val;
                        }
                    }
                    blockIndex++;
                }
            }

            blockIndex = 0;
            for (u32 block_y = 0; block_y < height / 2; block_y += 8) {
                for (u32 block_x = 0; block_x < width / 2; block_x += 8) {
                    //block level
                    int dx = motion_vectors_cb[blockIndex].dx;
                    int dy = motion_vectors_cb[blockIndex].dy;
                    for (u32 y = block_y; y < block_y + 8 && y < height / 2; y++) {
                        for (u32 x = block_x; x < block_x + 8 && x < width / 2; x++) {
                            int val = un_delta_Cb.at(y).at(x) + last_cb.at(y+dy).at(x+dx);
                            if (val > 255) {
                                val = 255;
                            }
                            else if (val < 0) {
                                val = 0;
                            }
                            Cb.at(y).at(x) = val;
                        }
                    }
                    blockIndex++;
                }
            }

            blockIndex = 0;
            for (u32 block_y = 0; block_y < height / 2; block_y += 8) {
                for (u32 block_x = 0; block_x < width / 2; block_x += 8) {
                    //block level
                    int dx = motion_vectors_cr[blockIndex].dx;
                    int dy = motion_vectors_cr[blockIndex].dy;
                    for (u32 y = block_y; y < block_y + 8 && y < height / 2; y++) {
                        for (u32 x = block_x; x < block_x + 8 && x < width / 2; x++) {
                            int val = un_delta_Cr.at(y).at(x) + last_cr.at(y+dy).at(x+dx);
                            if (val > 255) {
                                val = 255;
                            }
                            else if (val < 0) {
                                val = 0;
                            }
                            Cr.at(y).at(x) = val;
                        }
                    }
                    blockIndex++;
                }
            }


            /*
            std::clog << "P Frame reconstructed\n";
            print2DArray(Y); 
            */

            for (u32 y = 0; y < height; y++)
                for (u32 x = 0; x < width; x++)
                    frame.Y(x,y) = Y.at(y).at(x);
            for (u32 y = 0; y < height/2; y++)
                for (u32 x = 0; x < width/2; x++)
                    frame.Cb(x,y) = Cb.at(y).at(x);
            for (u32 y = 0; y < height/2; y++)
                for (u32 x = 0; x < width/2; x++)
                    frame.Cr(x,y) = Cr.at(y).at(x);
            writer.write_frame();


            last_frame.type = 1;
            last_frame.Y = Y;
            last_frame.Cb = Cb;
            last_frame.Cr = Cr;

            frame_type++;
            
        }
        frame_count++;

    }


    return 0;
}
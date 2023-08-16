//!-------------------------------------------------------------------------------------+
//! CSC485B - Assignment 4 - video compressor                                           +
//! Dylan Stevens                                                                       +
//! V00957595                                                                           +
//!-------------------------------------------------------------------------------------+

#include <iostream>
#include <fstream>
#include <cmath>
#include <array>
#include <string>
#include <cstring>
#include <cassert>
#include <cstdint>
#include <tuple>
#include "output_stream.hpp"
#include "yuv_stream.hpp"
#include <set>
#include <queue>
#include <functional>
#include <map>
#include <bitset>
#include <vector>
#include <algorithm>


#define BLK_SIZE 16


struct HuffmanNode {
    int symbol;
    unsigned int frequency;
    HuffmanNode* left;
    HuffmanNode* right;

    HuffmanNode(int sym, unsigned int freq) : symbol(sym), frequency(freq), left(nullptr), right(nullptr) {}
};

struct CompareNode {
    bool operator()(const HuffmanNode* lhs, const HuffmanNode* rhs) const {
        return lhs->frequency > rhs->frequency;
    }
};

struct HuffmanCode {
    unsigned int length;
    unsigned int code;
};


/**
 * @brief Prints the lengths of Huffman codes for each symbol.
 * 
 * @param codeLengths A map of symbol IDs to their corresponding code lengths.
 */
void printCodeLengths(const std::map<int, unsigned int>& codeLengths) {
    std::clog << "Symbol\tCode Length" << std::endl;
    for (const auto& entry : codeLengths) {
        int symbol = entry.first;
        unsigned int codeLength = entry.second;
        std::clog << symbol << "\t" << codeLength << std::endl;
    }
}


/**
 * @brief Creates a Huffman table from given bit lengths of symbols.
 * 
 * @param bitLengths A map of symbol IDs to their corresponding bit lengths.
 * @return The created Huffman table as a map of symbol IDs to code lengths.
 */
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


/**
 * @brief Generates static Huffman code lengths for symbols up to maxSymbol.
 * 
 * @param maxSymbol The maximum symbol ID.
 * @return A map of symbol IDs to generated code lengths.
 */
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

/**
 * @brief Generates dynamic Huffman code lengths based on input symbols.
 * 
 * @param symbols Input symbols for dynamic code generation.
 * @param maxSymbol The maximum symbol ID.
 * @return A map of symbol IDs to generated code lengths.
 */
std::map<int, unsigned int> generateDynamicCodeLengths(const std::vector<int>& symbols, int maxSymbol) {
    const int numSymbols = maxSymbol+2;  // Number of symbols, including special symbols


    //* Check if there is only one distinct element in the symbols vector
    bool singleSymbol = (std::set<int>(symbols.begin(), symbols.end()).size() == 1);

    //* Handle the single-symbol case separately
    if (singleSymbol) {
        int distinctSymbol = symbols[0];
        std::map<int, unsigned int> codeLengths;
        if (distinctSymbol >= -numSymbols && distinctSymbol < numSymbols) {
            codeLengths[distinctSymbol] = 1;
        }
        return codeLengths;
    }

    //* Count the frequency of each symbol
    std::map<int, unsigned int> frequencies;
    for (int symbol : symbols) {
        if (symbol >= -numSymbols && symbol < numSymbols) {
            frequencies[symbol]++;
        }
    }

    frequencies[maxSymbol + 1] = 1;

    //* Create Huffman tree nodes for each symbol
    std::priority_queue<HuffmanNode*, std::vector<HuffmanNode*>, CompareNode> pq;
    for (const auto& entry : frequencies) {
        int symbol = entry.first;
        unsigned int frequency = entry.second;
        HuffmanNode* node = new HuffmanNode(symbol, frequency);
        pq.push(node);
    }

    //* Build Huffman tree
    while (pq.size() > 1) {
        HuffmanNode* left = pq.top();
        pq.pop();
        HuffmanNode* right = pq.top();
        pq.pop();

        HuffmanNode* parent = new HuffmanNode(numSymbols, left->frequency + right->frequency);
        parent->left = left;
        parent->right = right;
        pq.push(parent);
    }

    //* Traverse the Huffman tree and assign code lengths
    std::map<int, unsigned int> codeLengths;
    if (!pq.empty()) {
        HuffmanNode* root = pq.top();

        //* Recursive function to assign code lengths
        std::function<void(HuffmanNode*, unsigned int)> assignCodeLengths = [&](HuffmanNode* node, unsigned int depth) {
            if (node->left == nullptr && node->right == nullptr) {
                if (node->symbol >= -numSymbols && node->symbol < numSymbols) {
                    codeLengths[node->symbol] = depth;
                }
            } else {
                if (node->left != nullptr) {
                    assignCodeLengths(node->left, depth + 1);
                }
                if (node->right != nullptr) {
                    assignCodeLengths(node->right, depth + 1);
                }
            }
        };

        assignCodeLengths(root, 0);
    }

    //* Cleanup
    while (!pq.empty()) {
        HuffmanNode* node = pq.top();
        pq.pop();
        delete node;
    }

    return codeLengths;

}



template<typename T>
std::vector<std::vector<T> > create_2d_vector(unsigned int outer, unsigned int inner) {
    std::vector<std::vector<T> > V {outer, std::vector<T>(inner,T() )};
    return V;
}

/**
 * @brief Creates a decompressed color plane using the specified quantization ID.
 * 
 * @param color_plane The color plane to decompress.
 * @param q_id The quantization ID.
 */
void createDecompressedColorPlane(std::vector<std::vector<unsigned char>>& color_plane, unsigned int q_id) {
    unsigned int height = color_plane.size();
    unsigned int width = color_plane[0].size();

    int m = BLK_SIZE;
    int n = BLK_SIZE;

    // Calculate the number of blocks in the image
    unsigned int num_blocks_x = (width + m - 1) / m;   // Round up the division
    unsigned int num_blocks_y = (height + n - 1) / n;  // Round up the division

    // Define the size of the DCT matrix
    int C_size = BLK_SIZE;

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
int Q_med[16][16] = {
    {32, 32, 20, 20, 20, 20, 32, 32, 48, 48, 80, 80, 100, 100, 120, 120},
    {32, 32, 20, 20, 20, 20, 32, 32, 48, 48, 80, 80, 100, 100, 120, 120},
    {24, 24, 28, 28, 36, 36, 52, 52, 116, 116, 120, 120, 108, 108, 108, 108},
    {24, 24, 28, 28, 36, 36, 52, 52, 116, 116, 120, 120, 108, 108, 108, 108},
    {28, 28, 24, 24, 32, 32, 48, 48, 80, 80, 112, 112, 136, 136, 112, 112},
    {28, 28, 24, 24, 32, 32, 48, 48, 80, 80, 112, 112, 136, 136, 112, 112},
    {28, 28, 32, 32, 44, 44, 56, 56, 100, 100, 172, 172, 160, 160, 124, 124},
    {28, 28, 32, 32, 44, 44, 56, 56, 100, 100, 172, 172, 160, 160, 124, 124},
    {36, 36, 44, 44, 72, 72, 112, 112, 136, 136, 216, 216, 204, 204, 152, 152},
    {36, 36, 44, 44, 72, 72, 112, 112, 136, 136, 216, 216, 204, 204, 152, 152},
    {48, 48, 68, 68, 108, 108, 128, 128, 160, 160, 208, 208, 224, 224, 184, 184},
    {48, 48, 68, 68, 108, 108, 128, 128, 160, 160, 208, 208, 224, 224, 184, 184},
    {96, 96, 128, 128, 156, 156, 172, 172, 204, 204, 240, 240, 240, 240, 200, 200},
    {96, 96, 128, 128, 156, 156, 172, 172, 204, 204, 240, 240, 240, 240, 200, 200},
    {144, 144, 184, 184, 188, 188, 196, 196, 224, 224, 200, 200, 204, 204, 196, 196},
    {144, 144, 184, 184, 188, 188, 196, 196, 224, 224, 200, 200, 204, 204, 196, 196}
};

int Q_hi[16][16] = {
    {8, 8, 5, 5, 5, 5, 8, 8, 12, 12, 20, 20, 25, 25, 30, 30},
    {8, 8, 5, 5, 5, 5, 8, 8, 12, 12, 20, 20, 25, 25, 30, 30},
    {6, 6, 6, 6, 7, 7, 9, 9, 13, 13, 29, 29, 30, 30, 27, 27},
    {6, 6, 6, 6, 7, 7, 9, 9, 13, 13, 29, 29, 30, 30, 27, 27},
    {7, 7, 6, 6, 8, 8, 12, 12, 20, 20, 28, 28, 34, 34, 28, 28},
    {7, 7, 6, 6, 8, 8, 12, 12, 20, 20, 28, 28, 34, 34, 28, 28},
    {7, 7, 8, 8, 11, 11, 14, 14, 25, 25, 43, 43, 40, 40, 31, 31},
    {7, 7, 8, 8, 11, 11, 14, 14, 25, 25, 43, 43, 40, 40, 31, 31},
    {9, 9, 11, 11, 18, 18, 28, 28, 34, 34, 54, 54, 51, 51, 38, 38},
    {9, 9, 11, 11, 18, 18, 28, 28, 34, 34, 54, 54, 51, 51, 38, 38},
    {12, 12, 17, 17, 27, 27, 32, 32, 40, 40, 52, 52, 56, 56, 46, 46},
    {12, 12, 17, 17, 27, 27, 32, 32, 40, 40, 52, 52, 56, 56, 46, 46},
    {24, 24, 32, 32, 39, 39, 43, 43, 51, 51, 60, 60, 60, 60, 50, 50},
    {24, 24, 32, 32, 39, 39, 43, 43, 51, 51, 60, 60, 60, 60, 50, 50},
    {36, 36, 46, 46, 47, 47, 49, 49, 56, 56, 50, 50, 51, 51, 49, 49},
    {36, 36, 46, 46, 47, 47, 49, 49, 56, 56, 50, 50, 51, 51, 49, 49}
};

int Q_low[16][16] = {
    {64, 64, 40, 40, 40, 40, 64, 64, 96, 96, 160, 160, 200, 200, 240, 240},
    {64, 64, 40, 40, 40, 40, 64, 64, 96, 96, 160, 160, 200, 200, 240, 240},
    {48, 48, 48, 48, 56, 56, 72, 72, 104, 104, 232, 232, 240, 240, 216, 216},
    {48, 48, 48, 48, 56, 56, 72, 72, 104, 104, 232, 232, 240, 240, 216, 216},
    {56, 56, 48, 48, 64, 64, 96, 96, 160, 160, 224, 224, 255, 255, 224, 224},
    {56, 56, 48, 48, 64, 64, 96, 96, 160, 160, 224, 224, 255, 255, 224, 224},
    {56, 56, 48, 48, 64, 64, 96, 96, 160, 160, 224, 224, 255, 255, 224, 224},
    {56, 56, 48, 48, 64, 64, 96, 96, 160, 160, 224, 224, 255, 255, 224, 224},
    {72, 72, 88, 88, 144, 144, 224, 224, 255, 255, 255, 255, 255, 255, 255, 255},
    {72, 72, 88, 88, 144, 144, 224, 224, 255, 255, 255, 255, 255, 255, 255, 255},
    {96, 96, 136, 136, 216, 216, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
    {96, 96, 136, 136, 216, 216, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
    {192, 192, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
    {192, 192, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
    {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
    {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
};


    // Iterate over each block in the image
    for (unsigned int block_y = 0; block_y < num_blocks_y; block_y++) {
        for (unsigned int block_x = 0; block_x < num_blocks_x; block_x++) {
            // Calculate the starting coordinates of the current block
            int start_x = block_x * m;
            int start_y = block_y * n;

            // Extract the 8x8 block from the input color plane
            float block[C_size][C_size];
            // Fill the temporary block with the color_plane data or duplicate the last rows/columns if needed
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    unsigned int y = start_y + i;
                    unsigned int x = start_x + j;

                    // If we reach the end of the height, duplicate the last row
                    if (y >= height) {
                        y = height - 1;
                    }
                    // If we reach the end of the width, duplicate the last column
                    if (x >= width) {
                        x = width - 1;
                    }

                    block[i][j] = color_plane[y][x];
                }
            }

            // Compute the matrix multiplication of (C)(block)
            float temp[C_size][C_size];
            memset(temp, 0, sizeof(temp));  // Initialize all elements of temp to zero
            for (int i = 0; i < C_size; i++) {
                for (int j = 0; j < C_size; j++) {
                    for (int k = 0; k < C_size; k++) {
                        temp[i][j] += C[i][k] * block[k][j];
                    }
                }
            }

            // Compute the matrix multiplication of (C)(block)(C^T)
            float dct_block[C_size][C_size];
            memset(dct_block, 0, sizeof(dct_block));  // Initialize all elements of dct_block to zero
            for (int i = 0; i < C_size; i++) {
                for (int j = 0; j < C_size; j++) {
                    for (int k = 0; k < C_size; k++) {
                        dct_block[i][j] += temp[i][k] * C_transpose[k][j];
                    }
                }
            }

            // Create the quantized DCT matrix
            int quantized_dct[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (q_id == 0) {
                        quantized_dct[i][j] = round(dct_block[i][j] / Q_low[i][j]);

                    } else if (q_id == 1) {
                        quantized_dct[i][j] = round(dct_block[i][j] / Q_med[i][j]);

                    } else if (q_id == 2) {
                        quantized_dct[i][j] = round(dct_block[i][j] / Q_hi[i][j]);
                    }
                }
            }

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
            memset(temp, 0, sizeof(temp));  // Initialize all elements of temp to zero
            for (int i = 0; i < C_size; i++) {
                for (int j = 0; j < C_size; j++) {
                    for (int k = 0; k < C_size; k++) {
                        temp[i][j] += C_transpose[i][k] * unquantized_dct[k][j];
                    }
                }
            }

            // Compute the matrix multiplication of (C^T)(block)(C)
            memset(dct_block, 0, sizeof(dct_block));  // Initialize all elements of dct_block to zero
            for (int i = 0; i < C_size; i++) {
                for (int j = 0; j < C_size; j++) {
                    for (int k = 0; k < C_size; k++) {
                        dct_block[i][j] += temp[i][k] * C[k][j];
                    }
                }
            }

            // Update the color_plane with the inverse DCT values
            for (int i = 0; i < m && (block_y * BLK_SIZE + i) < height; i++) {
                for (int j = 0; j < n && (block_x * BLK_SIZE + j) < width; j++) {
                    int rounded = std::round(dct_block[i][j]);
                    if (rounded < 0) {
                        color_plane.at(block_y * BLK_SIZE + i).at(block_x * BLK_SIZE + j) = 0;
                    } else if (rounded > 255) {
                        color_plane.at(block_y * BLK_SIZE + i).at(block_x * BLK_SIZE + j) = 255;
                    } else {
                        color_plane.at(block_y * BLK_SIZE + i).at(block_x * BLK_SIZE + j) = rounded;
                    }
                }
            }

        }
    }
}


/**
 * @brief Outputs prefix codes for given input using a Huffman table.
 * 
 * @param input The input data to be encoded.
 * @param huffmanTable The Huffman table for encoding.
 * @param bit_lengths_for_deltas Bit lengths for the delta encoded symbols.
 * @param run_len_symbol Symbol for run length encoding.
 * @param run_len_symbol_len Bit length of the run length symbol.
 * @return A vector of prefix codes as bits.
 */
std::vector<unsigned int> outputPrefixCodes(const std::vector<int>& input, const std::map<int, unsigned int>& huffmanTable, std::map<int, unsigned int> bit_lengths_for_deltas, unsigned int run_len_symbol, unsigned int run_len_symbol_len) {
    std::vector<unsigned int> data;
    int runs_less_than_8 = 0;
    int runs_over_8 = 0;
    unsigned int run_of_zeros = 0;

    for (int symbol : input) {
        auto it_huff = huffmanTable.find(symbol);
        auto it_bitlen = bit_lengths_for_deltas.find(symbol);
        if (it_huff != huffmanTable.end()) {
            unsigned int code = it_huff->second;
            unsigned int length;

            if (run_of_zeros == 0) {
                if (symbol == 0) {
                    run_of_zeros++;
                }
                else {
                    length = it_bitlen->second;
                    for (int j = length - 1; j >= 0; j--) {
                        unsigned int bit = (code >> j) & 1;
                        data.push_back(bit);
                        //std::clog << bit;
                    }
                }
            }
            else {
                if (symbol == 0) {
                    run_of_zeros++;
                }
                else {
                    if (run_of_zeros < 256) {
                        runs_less_than_8++;
                    }
                    else {
                        runs_over_8++;
                    }
                    if (run_of_zeros < (run_len_symbol_len+8)) {
                        length = bit_lengths_for_deltas.find(0)->second;
                        unsigned int zero_code = huffmanTable.find(0)->second;
                        for (unsigned int k = 0; k < run_of_zeros; k++) {
                            for (int j = length - 1; j >= 0; j--) {
                                unsigned int bit = (zero_code >> j) & 1;
                                data.push_back(bit);
                                //std::clog << bit;
                            }
                        }
                    }
                    else if (run_of_zeros >= (run_len_symbol_len+8)) {
                        
                        unsigned int run_code = run_len_symbol;
                        length = run_len_symbol_len;
                        //push the run symbol
                        for (int j = length - 1; j >= 0; j--) {
                            unsigned int bit = (run_code >> j) & 1;
                            data.push_back(bit);
                            //std::clog << bit;
                        }
                        
                        //push the run value as a u8
                        length = 8;
                        for (int j = length - 1; j >= 0; j--) {
                            unsigned int bit = (run_of_zeros >> j) & 1;
                            data.push_back(bit);
                            //std::clog << bit;
                        }
                        //std::clog<<std::endl;
                    }
                    length = it_bitlen->second;
                    //std::clog << "Code : " << code << "\n";
                    for (int j = length - 1; j >= 0; j--) {
                        unsigned int bit = (code >> j) & 1;
                        data.push_back(bit);
                        //std::clog << bit;
                    }
                    run_of_zeros = 0;
                }
                if (run_of_zeros == 255) {
                    unsigned int run_code = run_len_symbol;
                    length = run_len_symbol_len;
                    //push the run symbol
                    for (int j = length - 1; j >= 0; j--) {
                        unsigned int bit = (run_code >> j) & 1;
                        data.push_back(bit);
                        //std::clog << bit;
                    }
                    
                    //push the run value as a u8
                    length = 8;
                    for (int j = length - 1; j >= 0; j--) {
                        unsigned int bit = (run_of_zeros >> j) & 1;
                        data.push_back(bit);
                        //std::clog << bit;
                    }
                    run_of_zeros = 0;
                }
                
            }
        }
    }
    while (run_of_zeros > 0) {
        if (run_of_zeros >= (run_len_symbol_len+8)) {
            unsigned int run_code = run_len_symbol;
            unsigned length = run_len_symbol_len;
            //push the run symbol
            for (int j = length - 1; j >= 0; j--) {
                unsigned int bit = (run_code >> j) & 1;
                data.push_back(bit);
                //std::clog << bit;
            }
            //push the run value as a u8
            length = 8;
            for (int j = length - 1; j >= 0; j--) {
                unsigned int bit = (run_of_zeros >> j) & 1;
                data.push_back(bit);
                //std::clog << bit;
            }
        } else {
            // Run of length less than 9, encode individual zero symbols
            unsigned length = bit_lengths_for_deltas.find(0)->second;
            unsigned int zero_code = huffmanTable.find(0)->second;
            for (unsigned int k = 0; k < run_of_zeros; k++) {
                for (int j = length - 1; j >= 0; j--) {
                    unsigned int bit = (zero_code >> j) & 1;
                    data.push_back(bit);
                    //std::clog << bit;
                }
            }

        }
        run_of_zeros = 0;
    }
    //std::clog << "Runs less than 8: "<<runs_less_than_8<<", Runs over 8: "<<runs_over_8<<std::endl;
    return data;
}


/**
 * @brief Decodes Huffman-encoded data using a specified Huffman table.
 * 
 * @param bitstream The Huffman-encoded data as a bitstream.
 * @param huffmanTable The Huffman table for decoding.
 * @return Decoded data as a vector of integers.
 */
std::vector<int> decodeData(const std::vector<unsigned int>& bitstream, const std::map<int, unsigned int>& huffmanTable) {
    std::vector<int> decodedData;
    unsigned int numBits = 0;
    unsigned int currentCode = 0;

    std::map<unsigned int, int> reverseTable;
    for (const auto& entry : huffmanTable) {
        int symbol = entry.first;
        unsigned int code = entry.second;
        reverseTable[code] = symbol;
    }

    for (unsigned int bit : bitstream) {
        currentCode = (currentCode << 1) | bit;
        numBits++;

        auto it = reverseTable.find(currentCode);
        if (it != reverseTable.end()) {
            decodedData.push_back(it->second);
            numBits = 0;
            currentCode = 0;
        }
    }

    return decodedData;
}


/**
 * @brief Prints the contents of a Huffman table.
 * 
 * @param huffmanTable The Huffman table to be printed.
 */
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


/**
 * @brief Applies Delta DCT transformation to a frame.
 * 
 * @param frame_deltas The frame to be transformed.
 * @param stream The output bitstream to write the transformed data.
 * @param q_id The quantization ID.
 */
void Delta_DCT(std::vector<std::vector<int>>& frame_deltas, OutputBitStream stream, unsigned int q_id) {
    unsigned int height = frame_deltas.size();
    unsigned int width = frame_deltas[0].size();

    int m = BLK_SIZE;
    int n = BLK_SIZE;

    // Calculate the number of blocks in the image
    int num_blocks_x = (width + m - 1) / m;   // Round up the division
    int num_blocks_y = (height + n - 1) / n;  // Round up the division

    // Define the size of the DCT matrix
    int C_size = BLK_SIZE;

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
int Q_med[16][16] = {
    {32, 32, 20, 20, 20, 20, 32, 32, 48, 48, 80, 80, 100, 100, 120, 120},
    {32, 32, 20, 20, 20, 20, 32, 32, 48, 48, 80, 80, 100, 100, 120, 120},
    {24, 24, 28, 28, 36, 36, 52, 52, 116, 116, 120, 120, 108, 108, 108, 108},
    {24, 24, 28, 28, 36, 36, 52, 52, 116, 116, 120, 120, 108, 108, 108, 108},
    {28, 28, 24, 24, 32, 32, 48, 48, 80, 80, 112, 112, 136, 136, 112, 112},
    {28, 28, 24, 24, 32, 32, 48, 48, 80, 80, 112, 112, 136, 136, 112, 112},
    {28, 28, 32, 32, 44, 44, 56, 56, 100, 100, 172, 172, 160, 160, 124, 124},
    {28, 28, 32, 32, 44, 44, 56, 56, 100, 100, 172, 172, 160, 160, 124, 124},
    {36, 36, 44, 44, 72, 72, 112, 112, 136, 136, 216, 216, 204, 204, 152, 152},
    {36, 36, 44, 44, 72, 72, 112, 112, 136, 136, 216, 216, 204, 204, 152, 152},
    {48, 48, 68, 68, 108, 108, 128, 128, 160, 160, 208, 208, 224, 224, 184, 184},
    {48, 48, 68, 68, 108, 108, 128, 128, 160, 160, 208, 208, 224, 224, 184, 184},
    {96, 96, 128, 128, 156, 156, 172, 172, 204, 204, 240, 240, 240, 240, 200, 200},
    {96, 96, 128, 128, 156, 156, 172, 172, 204, 204, 240, 240, 240, 240, 200, 200},
    {144, 144, 184, 184, 188, 188, 196, 196, 224, 224, 200, 200, 204, 204, 196, 196},
    {144, 144, 184, 184, 188, 188, 196, 196, 224, 224, 200, 200, 204, 204, 196, 196}
};

int Q_hi[16][16] = {
    {8, 8, 5, 5, 5, 5, 8, 8, 12, 12, 20, 20, 25, 25, 30, 30},
    {8, 8, 5, 5, 5, 5, 8, 8, 12, 12, 20, 20, 25, 25, 30, 30},
    {6, 6, 6, 6, 7, 7, 9, 9, 13, 13, 29, 29, 30, 30, 27, 27},
    {6, 6, 6, 6, 7, 7, 9, 9, 13, 13, 29, 29, 30, 30, 27, 27},
    {7, 7, 6, 6, 8, 8, 12, 12, 20, 20, 28, 28, 34, 34, 28, 28},
    {7, 7, 6, 6, 8, 8, 12, 12, 20, 20, 28, 28, 34, 34, 28, 28},
    {7, 7, 8, 8, 11, 11, 14, 14, 25, 25, 43, 43, 40, 40, 31, 31},
    {7, 7, 8, 8, 11, 11, 14, 14, 25, 25, 43, 43, 40, 40, 31, 31},
    {9, 9, 11, 11, 18, 18, 28, 28, 34, 34, 54, 54, 51, 51, 38, 38},
    {9, 9, 11, 11, 18, 18, 28, 28, 34, 34, 54, 54, 51, 51, 38, 38},
    {12, 12, 17, 17, 27, 27, 32, 32, 40, 40, 52, 52, 56, 56, 46, 46},
    {12, 12, 17, 17, 27, 27, 32, 32, 40, 40, 52, 52, 56, 56, 46, 46},
    {24, 24, 32, 32, 39, 39, 43, 43, 51, 51, 60, 60, 60, 60, 50, 50},
    {24, 24, 32, 32, 39, 39, 43, 43, 51, 51, 60, 60, 60, 60, 50, 50},
    {36, 36, 46, 46, 47, 47, 49, 49, 56, 56, 50, 50, 51, 51, 49, 49},
    {36, 36, 46, 46, 47, 47, 49, 49, 56, 56, 50, 50, 51, 51, 49, 49}
};

int Q_low[16][16] = {
    {64, 64, 40, 40, 40, 40, 64, 64, 96, 96, 160, 160, 200, 200, 240, 240},
    {64, 64, 40, 40, 40, 40, 64, 64, 96, 96, 160, 160, 200, 200, 240, 240},
    {48, 48, 48, 48, 56, 56, 72, 72, 104, 104, 232, 232, 240, 240, 216, 216},
    {48, 48, 48, 48, 56, 56, 72, 72, 104, 104, 232, 232, 240, 240, 216, 216},
    {56, 56, 48, 48, 64, 64, 96, 96, 160, 160, 224, 224, 255, 255, 224, 224},
    {56, 56, 48, 48, 64, 64, 96, 96, 160, 160, 224, 224, 255, 255, 224, 224},
    {56, 56, 48, 48, 64, 64, 96, 96, 160, 160, 224, 224, 255, 255, 224, 224},
    {56, 56, 48, 48, 64, 64, 96, 96, 160, 160, 224, 224, 255, 255, 224, 224},
    {72, 72, 88, 88, 144, 144, 224, 224, 255, 255, 255, 255, 255, 255, 255, 255},
    {72, 72, 88, 88, 144, 144, 224, 224, 255, 255, 255, 255, 255, 255, 255, 255},
    {96, 96, 136, 136, 216, 216, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
    {96, 96, 136, 136, 216, 216, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
    {192, 192, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
    {192, 192, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
    {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
    {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
};

    
    std::vector<int> deltas;
    std::vector<u16> constants;

    // Iterate over each block in the frame
    for (int block_y = 0; block_y < num_blocks_y; block_y++) {
        for (int block_x = 0; block_x < num_blocks_x; block_x++) {
            // Calculate the starting coordinates of the current block
            int start_x = block_x * m;
            int start_y = block_y * n;

            // Extract the 8x8 block from the input color plane
            float block[C_size][C_size];
            // Fill the temporary block with the frame_deltas data or duplicate the last rows/columns if needed
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    unsigned int y = start_y + i;
                    unsigned int x = start_x + j;

                    // If we reach the end of the height, duplicate the last row
                    if (y >= height) {
                        y = height - 1;
                    }
                    // If we reach the end of the width, duplicate the last column
                    if (x >= width) {
                        x = width - 1;
                    }

                    block[i][j] = frame_deltas[y][x];
                }
            }

            // Compute the matrix multiplication of (C)(block)
            float temp[C_size][C_size];
            memset(temp, 0, sizeof(temp));  // Initialize all elements of temp to zero
            for (int i = 0; i < C_size; i++) {
                for (int j = 0; j < C_size; j++) {
                    for (int k = 0; k < C_size; k++) {
                        temp[i][j] += C[i][k] * block[k][j];
                    }
                }
            }

            // Compute the matrix multiplication of (C)(block)(C^T)
            float dct_block[C_size][C_size];
            memset(dct_block, 0, sizeof(dct_block));  // Initialize all elements of dct_block to zero
            for (int i = 0; i < C_size; i++) {
                for (int j = 0; j < C_size; j++) {
                    for (int k = 0; k < C_size; k++) {
                        dct_block[i][j] += temp[i][k] * C_transpose[k][j];
                    }
                }
            }

            // Create the quantized DCT matrix
            int quantized_dct[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (q_id == 0) {
                        quantized_dct[i][j] = round(dct_block[i][j] / Q_low[i][j]);

                    } else if (q_id == 1) {
                        quantized_dct[i][j] = round(dct_block[i][j] / Q_med[i][j]);

                    } else if (q_id == 2) {
                        quantized_dct[i][j] = round(dct_block[i][j] / Q_hi[i][j]);
                    }
                }
            }            

            int last_val = 0;
            int original_val = 0;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    int signed_val = (quantized_dct[i][j]);
                    if ((i == 0) && (j == 0 || j == 1)) {
                        original_val = signed_val;
                        u16 unsigned_casted_val = static_cast<u16>(signed_val);
                        constants.push_back(unsigned_casted_val);
                        last_val = signed_val;
                    } else {
                        original_val = signed_val;
                        signed_val = signed_val-last_val;
                        deltas.push_back(signed_val);
                        last_val = original_val;
                    }
                }
            }
        }
        
    }
    
    std::map<int, unsigned int> bit_lengths_for_deltas = generateDynamicCodeLengths(deltas, 512);

    std::map<int, unsigned int> huffmanTable = createHuffmanTable(bit_lengths_for_deltas);

    unsigned int run_len_symbol_len = bit_lengths_for_deltas[513];
    unsigned int run_len_symbol = huffmanTable[513];
    bit_lengths_for_deltas.erase(513);
    huffmanTable.erase(513);


    //push the code lengths
    int lowest = bit_lengths_for_deltas.begin()->first;
    int highest = bit_lengths_for_deltas.rbegin()->first;

    stream.push_u16(static_cast<u16>(lowest));
    stream.push_u16(static_cast<u16>(highest));
    
    for (int sym_iter = lowest; sym_iter <= highest; sym_iter++) {
        auto it = bit_lengths_for_deltas.find(sym_iter);
        if (it == bit_lengths_for_deltas.end()) {
            //push 0 bit length for symbol that doesn't exist
            //push in 4 bits
            unsigned int bitlen = 0;
            for (int j = 4; j >= 0; j--) {
                unsigned int v = (bitlen >> j) & 1;
                stream.push_bit(v);
            }
        } else {
            unsigned int bitlen = it->second;
            //push bit length for symbol that does exist
            for (int j = 4; j >= 0; j--) {
                unsigned int v = (bitlen >> j) & 1;
                stream.push_bit(v);
            }
        }
    }
    for (int j = 4; j >= 0; j--) {
        unsigned int v = (run_len_symbol_len >> j) & 1;
        stream.push_bit(v);
    }

    std::vector<unsigned int> bits = outputPrefixCodes(deltas, huffmanTable, bit_lengths_for_deltas, run_len_symbol, run_len_symbol_len);
    
    u32 num_of_bits = bits.size();
    stream.push_u32(num_of_bits);

    for (auto bit : bits) {
        stream.push_bit(bit);
    }

    for (auto constant : constants) {
        stream.push_u16(constant);
    }
    
    deltas.clear();
}

/**
 * @brief Applies Discrete Cosine Transformation (DCT) to a color plane.
 * 
 * @param color_plane The color plane to be transformed.
 * @param stream The output bitstream to write the transformed data.
 * @param q_id The quantization ID.
 */
void DCT(std::vector<std::vector<unsigned char>>& color_plane, OutputBitStream stream, unsigned int q_id) {
    unsigned int height = color_plane.size();
    unsigned int width = color_plane[0].size();

    int m = BLK_SIZE;
    int n = BLK_SIZE;

    // Calculate the number of blocks in the image
    int num_blocks_x = (width + m - 1) / m;   // Round up the division
    int num_blocks_y = (height + n - 1) / n;  // Round up the division

    // Define the size of the DCT matrix
    int C_size = BLK_SIZE;

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
int Q_med[16][16] = {
    {32, 32, 20, 20, 20, 20, 32, 32, 48, 48, 80, 80, 100, 100, 120, 120},
    {32, 32, 20, 20, 20, 20, 32, 32, 48, 48, 80, 80, 100, 100, 120, 120},
    {24, 24, 28, 28, 36, 36, 52, 52, 116, 116, 120, 120, 108, 108, 108, 108},
    {24, 24, 28, 28, 36, 36, 52, 52, 116, 116, 120, 120, 108, 108, 108, 108},
    {28, 28, 24, 24, 32, 32, 48, 48, 80, 80, 112, 112, 136, 136, 112, 112},
    {28, 28, 24, 24, 32, 32, 48, 48, 80, 80, 112, 112, 136, 136, 112, 112},
    {28, 28, 32, 32, 44, 44, 56, 56, 100, 100, 172, 172, 160, 160, 124, 124},
    {28, 28, 32, 32, 44, 44, 56, 56, 100, 100, 172, 172, 160, 160, 124, 124},
    {36, 36, 44, 44, 72, 72, 112, 112, 136, 136, 216, 216, 204, 204, 152, 152},
    {36, 36, 44, 44, 72, 72, 112, 112, 136, 136, 216, 216, 204, 204, 152, 152},
    {48, 48, 68, 68, 108, 108, 128, 128, 160, 160, 208, 208, 224, 224, 184, 184},
    {48, 48, 68, 68, 108, 108, 128, 128, 160, 160, 208, 208, 224, 224, 184, 184},
    {96, 96, 128, 128, 156, 156, 172, 172, 204, 204, 240, 240, 240, 240, 200, 200},
    {96, 96, 128, 128, 156, 156, 172, 172, 204, 204, 240, 240, 240, 240, 200, 200},
    {144, 144, 184, 184, 188, 188, 196, 196, 224, 224, 200, 200, 204, 204, 196, 196},
    {144, 144, 184, 184, 188, 188, 196, 196, 224, 224, 200, 200, 204, 204, 196, 196}
};

int Q_hi[16][16] = {
    {8, 8, 5, 5, 5, 5, 8, 8, 12, 12, 20, 20, 25, 25, 30, 30},
    {8, 8, 5, 5, 5, 5, 8, 8, 12, 12, 20, 20, 25, 25, 30, 30},
    {6, 6, 6, 6, 7, 7, 9, 9, 13, 13, 29, 29, 30, 30, 27, 27},
    {6, 6, 6, 6, 7, 7, 9, 9, 13, 13, 29, 29, 30, 30, 27, 27},
    {7, 7, 6, 6, 8, 8, 12, 12, 20, 20, 28, 28, 34, 34, 28, 28},
    {7, 7, 6, 6, 8, 8, 12, 12, 20, 20, 28, 28, 34, 34, 28, 28},
    {7, 7, 8, 8, 11, 11, 14, 14, 25, 25, 43, 43, 40, 40, 31, 31},
    {7, 7, 8, 8, 11, 11, 14, 14, 25, 25, 43, 43, 40, 40, 31, 31},
    {9, 9, 11, 11, 18, 18, 28, 28, 34, 34, 54, 54, 51, 51, 38, 38},
    {9, 9, 11, 11, 18, 18, 28, 28, 34, 34, 54, 54, 51, 51, 38, 38},
    {12, 12, 17, 17, 27, 27, 32, 32, 40, 40, 52, 52, 56, 56, 46, 46},
    {12, 12, 17, 17, 27, 27, 32, 32, 40, 40, 52, 52, 56, 56, 46, 46},
    {24, 24, 32, 32, 39, 39, 43, 43, 51, 51, 60, 60, 60, 60, 50, 50},
    {24, 24, 32, 32, 39, 39, 43, 43, 51, 51, 60, 60, 60, 60, 50, 50},
    {36, 36, 46, 46, 47, 47, 49, 49, 56, 56, 50, 50, 51, 51, 49, 49},
    {36, 36, 46, 46, 47, 47, 49, 49, 56, 56, 50, 50, 51, 51, 49, 49}
};

int Q_low[16][16] = {
    {64, 64, 40, 40, 40, 40, 64, 64, 96, 96, 160, 160, 200, 200, 240, 240},
    {64, 64, 40, 40, 40, 40, 64, 64, 96, 96, 160, 160, 200, 200, 240, 240},
    {48, 48, 48, 48, 56, 56, 72, 72, 104, 104, 232, 232, 240, 240, 216, 216},
    {48, 48, 48, 48, 56, 56, 72, 72, 104, 104, 232, 232, 240, 240, 216, 216},
    {56, 56, 48, 48, 64, 64, 96, 96, 160, 160, 224, 224, 255, 255, 224, 224},
    {56, 56, 48, 48, 64, 64, 96, 96, 160, 160, 224, 224, 255, 255, 224, 224},
    {56, 56, 48, 48, 64, 64, 96, 96, 160, 160, 224, 224, 255, 255, 224, 224},
    {56, 56, 48, 48, 64, 64, 96, 96, 160, 160, 224, 224, 255, 255, 224, 224},
    {72, 72, 88, 88, 144, 144, 224, 224, 255, 255, 255, 255, 255, 255, 255, 255},
    {72, 72, 88, 88, 144, 144, 224, 224, 255, 255, 255, 255, 255, 255, 255, 255},
    {96, 96, 136, 136, 216, 216, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
    {96, 96, 136, 136, 216, 216, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
    {192, 192, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
    {192, 192, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
    {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
    {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
};

    
    std::vector<int> deltas;
    std::vector<u16> constants;

    // Iterate over each block in the frame
    for (int block_y = 0; block_y < num_blocks_y; block_y++) {
        for (int block_x = 0; block_x < num_blocks_x; block_x++) {
            // Calculate the starting coordinates of the current block
            int start_x = block_x * m;
            int start_y = block_y * n;

            // Extract the 8x8 block from the input color plane
            float block[C_size][C_size];
            // Fill the temporary block with the color_plane data or duplicate the last rows/columns if needed
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    unsigned int y = start_y + i;
                    unsigned int x = start_x + j;

                    // If we reach the end of the height, duplicate the last row
                    if (y >= height) {
                        y = height - 1;
                    }
                    // If we reach the end of the width, duplicate the last column
                    if (x >= width) {
                        x = width - 1;
                    }

                    block[i][j] = color_plane[y][x];
                }
            }

            // Compute the matrix multiplication of (C)(block)
            float temp[C_size][C_size];
            memset(temp, 0, sizeof(temp));  // Initialize all elements of temp to zero
            for (int i = 0; i < C_size; i++) {
                for (int j = 0; j < C_size; j++) {
                    for (int k = 0; k < C_size; k++) {
                        temp[i][j] += C[i][k] * block[k][j];
                    }
                }
            }

            // Compute the matrix multiplication of (C)(block)(C^T)
            float dct_block[C_size][C_size];
            memset(dct_block, 0, sizeof(dct_block));  // Initialize all elements of dct_block to zero
            for (int i = 0; i < C_size; i++) {
                for (int j = 0; j < C_size; j++) {
                    for (int k = 0; k < C_size; k++) {
                        dct_block[i][j] += temp[i][k] * C_transpose[k][j];
                    }
                }
            }

            // Create the quantized DCT matrix
            int quantized_dct[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (q_id == 0) {
                        quantized_dct[i][j] = round(dct_block[i][j] / Q_low[i][j]);

                    } else if (q_id == 1) {
                        quantized_dct[i][j] = round(dct_block[i][j] / Q_med[i][j]);

                    } else if (q_id == 2) {
                        quantized_dct[i][j] = round(dct_block[i][j] / Q_hi[i][j]);
                    }
                }
            }  

            int last_val = 0;
            int original_val = 0;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    int signed_val = (quantized_dct[i][j]);
                    if ((i == 0) && (j == 0 || j == 1)) {
                        original_val = signed_val;
                        u16 unsigned_casted_val = static_cast<u16>(signed_val);
                        constants.push_back(unsigned_casted_val);
                        last_val = signed_val;
                    } else {
                        original_val = signed_val;
                        signed_val = signed_val-last_val;
                        deltas.push_back(signed_val);
                        last_val = original_val;
                    }
                }
            }
        }
        
    }

    std::map<int, unsigned int> bit_lengths_for_deltas = generateDynamicCodeLengths(deltas, 512);

    std::map<int, unsigned int> huffmanTable = createHuffmanTable(bit_lengths_for_deltas);

    unsigned int run_len_symbol_len = bit_lengths_for_deltas[513];
    unsigned int run_len_symbol = huffmanTable[513];

    bit_lengths_for_deltas.erase(513);
    huffmanTable.erase(513);

    //push the code lengths
    int lowest = bit_lengths_for_deltas.begin()->first;
    int highest = bit_lengths_for_deltas.rbegin()->first;

    stream.push_u16(static_cast<u16>(lowest));
    stream.push_u16(static_cast<u16>(highest));
    
    for (int sym_iter = lowest; sym_iter <= highest; sym_iter++) {
        auto it = bit_lengths_for_deltas.find(sym_iter);
        if (it == bit_lengths_for_deltas.end()) {
            //push 0 bit length for symbol that doesn't exist
            //push in 4 bits
            unsigned int bitlen = 0;
            for (int j = 4; j >= 0; j--) {
                unsigned int v = (bitlen >> j) & 1;
                stream.push_bit(v);
            }
        } else {
            unsigned int bitlen = it->second;
            //push bit length for symbol that does exist
            for (int j = 4; j >= 0; j--) {
                unsigned int v = (bitlen >> j) & 1;
                stream.push_bit(v);
            }
        }
    }
    for (int j = 4; j >= 0; j--) {
        unsigned int v = (run_len_symbol_len >> j) & 1;
        stream.push_bit(v);
    }
    
    std::vector<unsigned int> bits = outputPrefixCodes(deltas, huffmanTable, bit_lengths_for_deltas, run_len_symbol, run_len_symbol_len);
    
    u32 num_of_bits = bits.size();
    stream.push_u32(num_of_bits);

    for (auto bit : bits) {
        stream.push_bit(bit);
    }

    for (auto constant : constants) {
        stream.push_u16(constant);
    }

    deltas.clear();
}


/**
 * @brief Prints the contents of a 2D array.
 * 
 * @param array The 2D array to be printed.
 */
void print2DArray(const std::vector<std::vector<unsigned char>>& array) {
    for (int i = 0; i < BLK_SIZE; i++) {
        for (int j = 0; j < BLK_SIZE; j++) {
            std::clog << static_cast<unsigned int>(array.at(i).at(j)) << "    ";
        }
        std::clog << std::endl;
    }
    std::clog << std::endl;
}



/**
 * @brief Prints the contents of a 2D delta array.
 * 
 * @param array The 2D delta array to be printed.
 */
void print2DdeltaArray(const std::vector<std::vector<int>>& array) {
    for (int i = 0; i < BLK_SIZE; i++) {
        for (int j = 0; j < BLK_SIZE; j++) {
            std::clog << (array.at(i).at(j)) << "    ";
        }
        std::clog << std::endl;
    }
    std::clog << std::endl;
}



struct Frame_track {
    unsigned int type;
    // 1 for I frame, 0 for P frame
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


struct MotionVector {
    int dx; // Horizontal motion vector
    int dy; // Vertical motion vector
};


/**
 * @brief Calculates motion vectors for a video frame.
 * 
 * @param currentFrame The current video frame.
 * @param referenceFrame The reference frame for comparison.
 * @param blockWidth The width of motion estimation blocks.
 * @param blockHeight The height of motion estimation blocks.
 * @param searchRange The search range for motion estimation.
 * @return A vector of calculated motion vectors.
 */
std::vector<MotionVector> calculateMotionVectors(
    const std::vector<std::vector<unsigned char>>& currentFrame,
    const std::vector<std::vector<unsigned char>>& referenceFrame,
    int blockWidth, int blockHeight, int searchRange) {

    std::vector<MotionVector> motionVectors;

    int width = currentFrame[0].size();
    int height = currentFrame.size();

    for (int y = 0; y < height; y += blockHeight) {
        for (int x = 0; x < width; x += blockWidth) {
            MotionVector bestMotionVector = {0, 0}; // Initialize the best motion vector
            int minSAD = std::numeric_limits<int>::max(); // Initialize the minimum sum of absolute differences

            for (int dy = -searchRange; dy <= searchRange; ++dy) {
                for (int dx = -searchRange; dx <= searchRange; ++dx) {
                    int SAD = 0; // Initialize the sum of absolute differences
                    
                    // Calculate the reference block position
                    int refX = x + dx;
                    int refY = y + dy;

                    // Check if the reference block position is within bounds
                    if (refX >= 0 && refX + blockWidth <= width && refY >= 0 && refY + blockHeight <= height) {
                        for (int by = 0; by < blockHeight; ++by) {
                            for (int bx = 0; bx < blockWidth; ++bx) {
                                int diff = static_cast<int>(currentFrame[y + by][x + bx]) - static_cast<int>(referenceFrame[refY + by][refX + bx]);
                                SAD += std::abs(diff);
                            }
                        }

                        if (SAD < minSAD) {
                            minSAD = SAD;
                            bestMotionVector.dx = dx;
                            bestMotionVector.dy = dy;
                        }
                    }
                }
            }
            motionVectors.push_back(bestMotionVector);
        }
    }

    return motionVectors;
}


/**
 * @brief Prints motion vectors for a video frame.
 * 
 * @param motionVectors The motion vectors to be printed.
 * @param blockWidth The width of motion estimation blocks.
 * @param blockHeight The height of motion estimation blocks.
 * @param frameWidth The width of the video frame.
 */
void printMotionVectors(const std::vector<MotionVector>& motionVectors, int blockWidth, int blockHeight, int frameWidth) {
    for (unsigned int blockIndex = 0; blockIndex < motionVectors.size(); ++blockIndex) {
        int blockX = (blockIndex % (frameWidth / blockWidth)) * blockWidth;
        int blockY = (blockIndex / (frameWidth / blockWidth)) * blockHeight;

        std::clog << "Block (" << blockX << ", " << blockY << ") - Motion Vector: (" << motionVectors[blockIndex].dx << ", " << motionVectors[blockIndex].dy << ")\n";
    }
}


/**
* @brief Main driver function
*/
int main(int argc, char** argv){

    if (argc < 4){
        std::clog << "Usage: " << argv[0] << " <width> <height> <low/medium/high>" << std::endl;
        return 1;
    }
    u32 width = std::stoi(argv[1]);
    u32 height = std::stoi(argv[2]);
    std::string quality{argv[3]};

    unsigned int q_id;

    if (quality == "low") {
        q_id = 0;
    } else if (quality == "medium") {
        q_id = 1;
    } else if (quality == "high") {
        q_id = 2;
    } else {
        std::clog << "Usage: " << argv[0] << " <width> <height> <low/medium/high>" << std::endl;
        return 1;
    }

    YUVStreamReader reader {std::cin, width, height};
    OutputBitStream output_stream {std::cout};

    output_stream.push_u32(height);
    output_stream.push_u32(width);

    output_stream.push_byte(q_id);

    Frame_track last_frame(1, {{0}}, {{0}}, {{0}});

    int frame_type = 4;

    int frame_count = 0;

    while (reader.read_next_frame()){
        output_stream.push_byte(1); //Use a one byte flag to indicate whether there is a frame here
        YUVFrame420& frame = reader.frame();

        //extract Y values
        auto Y = create_2d_vector<unsigned char>(height, width);

        //extract Cb values
        auto Cb = create_2d_vector<unsigned char>(height/2, width/2);
        

        //extract Cr values
        auto Cr = create_2d_vector<unsigned char>(height/2, width/2);
                
        //I Frame
        if (frame_type == 4) {
            for (u32 y = 0; y < height; y++)
                for (u32 x = 0; x < width; x++)
                    Y.at(y).at(x) = frame.Y(x,y);

            for (u32 y = 0; y < height/2; y++)
                for (u32 x = 0; x < width/2; x++)
                    Cb.at(y).at(x) = frame.Cb(x,y);
            
            for (u32 y = 0; y < height/2; y++)
                for (u32 x = 0; x < width/2; x++)
                    Cr.at(y).at(x) = frame.Cr(x,y);

            DCT(Y, output_stream, q_id);
            DCT(Cb, output_stream, q_id);
            DCT(Cr, output_stream, q_id);
            
            last_frame.type = 0;
            last_frame.Y = Y;
            last_frame.Cb = Cb;
            last_frame.Cr = Cr;

            frame_type = 0;   
        }

        //P Frame
        else if (frame_type < 4) {
        
            //Get the old values and compressed, then compress them (necessary so that the decompressor has access to the matrix)
            auto last_y = last_frame.Y;
            auto last_cb = last_frame.Cb;
            auto last_cr = last_frame.Cr;
            
            createDecompressedColorPlane(last_y, q_id);
            createDecompressedColorPlane(last_cb, q_id);
            createDecompressedColorPlane(last_cr, q_id);

            for (u32 y = 0; y < height; y++)
                for (u32 x = 0; x < width; x++)
                    Y.at(y).at(x) = frame.Y(x,y);

            for (u32 y = 0; y < height/2; y++)
                for (u32 x = 0; x < width/2; x++)
                    Cb.at(y).at(x) = frame.Cb(x,y);
            
            for (u32 y = 0; y < height/2; y++)
                for (u32 x = 0; x < width/2; x++)
                    Cr.at(y).at(x) = frame.Cr(x,y);

            auto delta_Y = create_2d_vector<int>(height, width);
            auto delta_Cb = create_2d_vector<int>(height/2, width/2);
            auto delta_Cr = create_2d_vector<int>(height/2, width/2);

            std::vector<MotionVector> motion_vectors_y = calculateMotionVectors(Y, last_y, BLK_SIZE, BLK_SIZE, BLK_SIZE);
            std::vector<MotionVector> motion_vectors_cb = calculateMotionVectors(Cb, last_y, BLK_SIZE, BLK_SIZE, BLK_SIZE);
            std::vector<MotionVector> motion_vectors_cr = calculateMotionVectors(Cr, last_y, BLK_SIZE, BLK_SIZE, BLK_SIZE);

            int numbitspushed = 0;
            for (u32 i = 0; i < motion_vectors_y.size(); i++) {
                int dx = motion_vectors_y[i].dx;
                if (dx < 0) {
                    output_stream.push_bit(0);
                }
                else {
                    output_stream.push_bit(1);
                }
                numbitspushed++;
                for (int j = 4; j >= 0; j--) {
                    unsigned int bit = (abs(dx) >> j) & 1;
                    output_stream.push_bit(bit);
                    numbitspushed++;
                }

                int dy = motion_vectors_y[i].dy;
                if (dy < 0) {
                    output_stream.push_bit(0);
                }
                else {
                    output_stream.push_bit(1);
                }
                numbitspushed++;
                for (int j = 4; j >= 0; j--) {
                    unsigned int bit = (abs(dy) >> j) & 1;
                    output_stream.push_bit(bit);
                    numbitspushed++;
                }
            }

            //push cb motion vectors
            for (u32 i = 0; i < motion_vectors_cb.size(); i++) {
                int dx = motion_vectors_cb[i].dx;
                if (dx < 0) {
                    output_stream.push_bit(0);
                }
                else {
                    output_stream.push_bit(1);
                }
                numbitspushed++;
                for (int j = 4; j >= 0; j--) {
                    unsigned int bit = (abs(dx) >> j) & 1;
                    output_stream.push_bit(bit);
                                    numbitspushed++;

                }

                int dy = motion_vectors_cb[i].dy;
                if (dy < 0) {
                    output_stream.push_bit(0);
                }
                else {
                    output_stream.push_bit(1);
                }
                                numbitspushed++;

                for (int j = 4; j >= 0; j--) {
                    unsigned int bit = (abs(dy) >> j) & 1;
                    output_stream.push_bit(bit);
                                    numbitspushed++;

                }
            }

            //push cr motion vectors
            for (u32 i = 0; i < motion_vectors_cr.size(); i++) {
                int dx = motion_vectors_cr[i].dx;
                if (dx < 0) {
                    output_stream.push_bit(0);
                }
                else {
                    output_stream.push_bit(1);
                }
                                numbitspushed++;

                for (int j = 4; j >= 0; j--) {
                    unsigned int bit = (abs(dx) >> j) & 1;
                    output_stream.push_bit(bit);
                                    numbitspushed++;

                }
                
                int dy = motion_vectors_cr[i].dy;
                if (dy < 0) {
                    output_stream.push_bit(0);
                }
                else {
                    output_stream.push_bit(1);
                }
                                numbitspushed++;

                for (int j = 4; j >= 0; j--) {
                    unsigned int bit = (abs(dy) >> j) & 1;
                    output_stream.push_bit(bit);
                                    numbitspushed++;

                }
            }
            
            unsigned int blockIndex = 0;
            //frame level
            for (u32 block_y = 0; block_y < height; block_y += BLK_SIZE) {
                for (u32 block_x = 0; block_x < width; block_x += BLK_SIZE) {
                    //block level
                    int dx = motion_vectors_y[blockIndex].dx;
                    int dy = motion_vectors_y[blockIndex].dy;
                    for (u32 y = block_y; y < block_y + BLK_SIZE && y < height; y++) {
                        for (u32 x = block_x; x < block_x + BLK_SIZE && x < width; x++) {
                            //calculate the difference using the motion vectors
                            delta_Y.at(y).at(x) = frame.Y(x, y) - last_y.at(y+dy).at(x+dx);
                            
                        }
                    }
                    blockIndex++;
                }
            }

            blockIndex = 0;
            //frame level
            for (u32 block_y = 0; block_y < height / 2; block_y += BLK_SIZE) {
                for (u32 block_x = 0; block_x < width / 2; block_x += BLK_SIZE) {
                    //block level
                    int dx = motion_vectors_cb[blockIndex].dx;
                    int dy = motion_vectors_cb[blockIndex].dy;
                    for (u32 y = block_y; y < block_y + BLK_SIZE && y < height / 2; y++) {
                        for (u32 x = block_x; x < block_x + BLK_SIZE && x < width / 2; x++) {
                            delta_Cb.at(y).at(x) = frame.Cb(x, y) - last_cb.at(y+dy).at(x+dx);
                        }
                    }
                    blockIndex++;
                }
            }

            blockIndex = 0;
            //frame level
            for (u32 block_y = 0; block_y < height / 2; block_y += BLK_SIZE) {
                for (u32 block_x = 0; block_x < width / 2; block_x += BLK_SIZE) {
                    //block level
                    int dx = motion_vectors_cr[blockIndex].dx;
                    int dy = motion_vectors_cr[blockIndex].dy;
                    for (u32 y = block_y; y < block_y + BLK_SIZE && y < height / 2; y++) {
                        for (u32 x = block_x; x < block_x + BLK_SIZE && x < width / 2; x++) {
                            delta_Cr.at(y).at(x) = frame.Cr(x, y) - last_cr.at(y+dy).at(x+dx);
                        }
                    }
                    blockIndex++;
                }
            }

            Delta_DCT(delta_Y, output_stream, q_id);
            Delta_DCT(delta_Cb, output_stream, q_id);
            Delta_DCT(delta_Cr, output_stream, q_id);

            last_frame.type = 1;
            last_frame.Y = Y;
            last_frame.Cb = Cb;
            last_frame.Cr = Cr;

            frame_type++;
        }  
        frame_count++;      
    }
    output_stream.push_byte(0); //Flag to indicate end of data
    output_stream.flush_to_byte();

    return 0;
}
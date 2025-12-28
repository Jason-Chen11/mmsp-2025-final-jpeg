// MMSP 2025 Final Project - JPEG Encoder
// Standalone version with all dependencies included

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ========== Header Definitions ==========

#ifndef BMP_H
#define BMP_H

#include <stdint.h>

#pragma pack(push, 1)
typedef struct {
    uint16_t type;
    uint32_t size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;
} BMPHeader;

typedef struct {
    uint32_t size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bits;
    uint32_t compression;
    uint32_t imagesize;
    int32_t xresolution;
    int32_t yresolution;
    uint32_t ncolours;
    uint32_t importantcolours;
} BMPInfoHeader;
#pragma pack(pop)

typedef struct {
    int width;
    int height;
    uint8_t *R;
    uint8_t *G;
    uint8_t *B;
    uint8_t *header_data;
    int header_size;
} Image;

Image* read_bmp(const char *filename);
void write_bmp(const char *filename, Image *img);
void free_image(Image *img);

#endif

#ifndef JPEG_UTILS_H
#define JPEG_UTILS_H

#include <stdint.h>

void rgb_to_ycbcr(Image *img, double **Y, double **Cb, double **Cr);
void ycbcr_to_rgb(Image *img, double **Y, double **Cb, double **Cr);

void dct_2d(double block[8][8]);
void idct_2d(double block[8][8]);

void quantize(double **F, int16_t **qF, int width, int height, const int Qt[8][8]);
void dequantize(int16_t **qF, double **F, int width, int height, const int Qt[8][8]);

void calculate_quantization_error(double **F, int16_t **qF, float **eF, 
                                  int width, int height, const int Qt[8][8]);

void zigzag_scan(int16_t block[8][8], int16_t *output);
void inverse_zigzag_scan(int16_t *input, int16_t block[8][8]);

typedef struct {
    int skip;
    int16_t value;
} RLEPair;

typedef struct {
    RLEPair *pairs;
    int count;
    int capacity;
} RLECode;

RLECode* rle_encode(int16_t *zigzag, int length);
void rle_decode(RLECode *rle, int16_t *output);
void free_rle(RLECode *rle);

typedef struct {
    int16_t symbol;
    int count;
    char *codeword;
} HuffmanSymbol;

typedef struct HuffmanNode {
    int16_t symbol;
    int count;
    struct HuffmanNode *left;
    struct HuffmanNode *right;
} HuffmanNode;

typedef struct {
    HuffmanSymbol *symbols;
    int count;
} Codebook;

Codebook* build_codebook(int16_t *symbols, int length);
char* huffman_encode(int16_t *symbols, int length, Codebook *codebook);
int16_t* huffman_decode(const char *bitstream, Codebook *codebook, int *out_length);
void free_codebook(Codebook *cb);

double calculate_sqnr(double *signal, double *error, int length);
double calculate_sqnr_uint8(uint8_t *original, uint8_t *reconstructed, int length);

extern const int Qt_Luminance[8][8];
extern const int Qt_Chrominance[8][8];
extern const int ZigZag[64];

#endif


// ========== BMP Functions ==========

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Image* read_bmp(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        return NULL;
    }

    fseek(f, 0, SEEK_SET);
    
    BMPHeader header;
    BMPInfoHeader info;
    
    fread(&header, sizeof(BMPHeader), 1, f);
    fread(&info, sizeof(BMPInfoHeader), 1, f);

    if (header.type != 0x4D42) {
        fprintf(stderr, "Not a BMP file\n");
        fclose(f);
        return NULL;
    }

    if (info.bits != 24) {
        fprintf(stderr, "Only 24-bit BMP supported\n");
        fclose(f);
        return NULL;
    }

    Image *img = (Image*)malloc(sizeof(Image));
    img->width = info.width;
    img->height = info.height;
    img->header_size = header.offset;
    
    img->header_data = (uint8_t*)malloc(img->header_size);
    fseek(f, 0, SEEK_SET);
    fread(img->header_data, 1, img->header_size, f);
    
    int size = img->width * img->height;
    img->R = (uint8_t*)malloc(size);
    img->G = (uint8_t*)malloc(size);
    img->B = (uint8_t*)malloc(size);

    fseek(f, header.offset, SEEK_SET);

    int padding = (4 - (img->width * 3) % 4) % 4;
    
    for (int y = img->height - 1; y >= 0; y--) {
        for (int x = 0; x < img->width; x++) {
            int idx = y * img->width + x;
            img->B[idx] = fgetc(f);
            img->G[idx] = fgetc(f);
            img->R[idx] = fgetc(f);
        }
        fseek(f, padding, SEEK_CUR);
    }

    fclose(f);
    return img;
}

void write_bmp(const char *filename, Image *img) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Cannot create file %s\n", filename);
        return;
    }

    fwrite(img->header_data, 1, img->header_size, f);

    int padding = (4 - (img->width * 3) % 4) % 4;
    uint8_t pad[3] = {0, 0, 0};
    
    for (int y = img->height - 1; y >= 0; y--) {
        for (int x = 0; x < img->width; x++) {
            int idx = y * img->width + x;
            fputc(img->B[idx], f);
            fputc(img->G[idx], f);
            fputc(img->R[idx], f);
        }
        fwrite(pad, 1, padding, f);
    }

    fclose(f);
}

void free_image(Image *img) {
    if (img) {
        free(img->R);
        free(img->G);
        free(img->B);
        free(img->header_data);
        free(img);
    }
}


// ========== JPEG Utils ==========

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

const int Qt_Luminance[8][8] = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
};

const int Qt_Chrominance[8][8] = {
    {17, 18, 24, 47, 99, 99, 99, 99},
    {18, 21, 26, 66, 99, 99, 99, 99},
    {24, 26, 56, 99, 99, 99, 99, 99},
    {47, 66, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99}
};

const int ZigZag[64] = {
    0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

void rgb_to_ycbcr(Image *img, double **Y, double **Cb, double **Cr) {
    int size = img->width * img->height;
    *Y = (double*)malloc(size * sizeof(double));
    *Cb = (double*)malloc(size * sizeof(double));
    *Cr = (double*)malloc(size * sizeof(double));

    for (int i = 0; i < size; i++) {
        double r = img->R[i];
        double g = img->G[i];
        double b = img->B[i];

        (*Y)[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        (*Cb)[i] = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
        (*Cr)[i] = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;
    }
}

void ycbcr_to_rgb(Image *img, double **Y, double **Cb, double **Cr) {
    int size = img->width * img->height;

    for (int i = 0; i < size; i++) {
        double y = (*Y)[i];
        double cb = (*Cb)[i] - 128.0;
        double cr = (*Cr)[i] - 128.0;

        double r = y + 1.402 * cr;
        double g = y - 0.344136 * cb - 0.714136 * cr;
        double b = y + 1.772 * cb;

        img->R[i] = (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
        img->G[i] = (uint8_t)(g < 0 ? 0 : (g > 255 ? 255 : g));
        img->B[i] = (uint8_t)(b < 0 ? 0 : (b > 255 ? 255 : b));
    }
}

void dct_2d(double block[8][8]) {
    double temp[8][8];
    const double pi = 3.14159265358979323846;

    for (int u = 0; u < 8; u++) {
        for (int v = 0; v < 8; v++) {
            double sum = 0.0;
            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    sum += (block[x][y] - 128.0) * 
                           cos((2*x + 1) * u * pi / 16.0) *
                           cos((2*y + 1) * v * pi / 16.0);
                }
            }
            double cu = (u == 0) ? 1.0/sqrt(2.0) : 1.0;
            double cv = (v == 0) ? 1.0/sqrt(2.0) : 1.0;
            temp[u][v] = 0.25 * cu * cv * sum;
        }
    }

    memcpy(block, temp, sizeof(temp));
}

void idct_2d(double block[8][8]) {
    double temp[8][8];
    const double pi = 3.14159265358979323846;

    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            double sum = 0.0;
            for (int u = 0; u < 8; u++) {
                for (int v = 0; v < 8; v++) {
                    double cu = (u == 0) ? 1.0/sqrt(2.0) : 1.0;
                    double cv = (v == 0) ? 1.0/sqrt(2.0) : 1.0;
                    sum += cu * cv * block[u][v] *
                           cos((2*x + 1) * u * pi / 16.0) *
                           cos((2*y + 1) * v * pi / 16.0);
                }
            }
            temp[x][y] = 0.25 * sum + 128.0;
        }
    }

    memcpy(block, temp, sizeof(temp));
}

void quantize(double **F, int16_t **qF, int width, int height, const int Qt[8][8]) {
    int blocks_w = width / 8;
    int blocks_h = height / 8;

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    (*qF)[idx] = (int16_t)round((*F)[idx] / Qt[i][j]);
                }
            }
        }
    }
}

void dequantize(int16_t **qF, double **F, int width, int height, const int Qt[8][8]) {
    int blocks_w = width / 8;
    int blocks_h = height / 8;

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    (*F)[idx] = (*qF)[idx] * Qt[i][j];
                }
            }
        }
    }
}

void calculate_quantization_error(double **F, int16_t **qF, float **eF, 
                                  int width, int height, const int Qt[8][8]) {
    int blocks_w = width / 8;
    int blocks_h = height / 8;

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    (*eF)[idx] = (float)((*F)[idx] - (*qF)[idx] * Qt[i][j]);
                }
            }
        }
    }
}

double calculate_sqnr(double *signal, double *error, int length) {
    double signal_power = 0.0;
    double error_power = 0.0;

    for (int i = 0; i < length; i++) {
        signal_power += signal[i] * signal[i];
        error_power += error[i] * error[i];
    }

    if (error_power < 1e-10) return 100.0;
    return 10.0 * log10(signal_power / error_power);
}

double calculate_sqnr_uint8(uint8_t *original, uint8_t *reconstructed, int length) {
    double signal_power = 0.0;
    double error_power = 0.0;

    for (int i = 0; i < length; i++) {
        double s = (double)original[i];
        double e = (double)original[i] - (double)reconstructed[i];
        signal_power += s * s;
        error_power += e * e;
    }

    if (error_power < 1e-10) return 100.0;
    return 10.0 * log10(signal_power / error_power);
}


// ========== RLE & Huffman ==========

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void zigzag_scan(int16_t block[8][8], int16_t *output) {
    for (int i = 0; i < 64; i++) {
        int pos = ZigZag[i];
        int row = pos / 8;
        int col = pos % 8;
        output[i] = block[row][col];
    }
}

void inverse_zigzag_scan(int16_t *input, int16_t block[8][8]) {
    for (int i = 0; i < 64; i++) {
        int pos = ZigZag[i];
        int row = pos / 8;
        int col = pos % 8;
        block[row][col] = input[i];
    }
}

RLECode* rle_encode(int16_t *zigzag, int length) {
    RLECode *rle = (RLECode*)malloc(sizeof(RLECode));
    rle->capacity = 128;
    rle->pairs = (RLEPair*)malloc(rle->capacity * sizeof(RLEPair));
    rle->count = 0;

    int i = 1;
    while (i < length) {
        int skip = 0;
        while (i < length && zigzag[i] == 0) {
            skip++;
            i++;
        }

        if (i < length) {
            if (rle->count >= rle->capacity) {
                rle->capacity *= 2;
                rle->pairs = (RLEPair*)realloc(rle->pairs, rle->capacity * sizeof(RLEPair));
            }
            rle->pairs[rle->count].skip = skip;
            rle->pairs[rle->count].value = zigzag[i];
            rle->count++;
            i++;
        } else {
            if (rle->count >= rle->capacity) {
                rle->capacity *= 2;
                rle->pairs = (RLEPair*)realloc(rle->pairs, rle->capacity * sizeof(RLEPair));
            }
            rle->pairs[rle->count].skip = 0;
            rle->pairs[rle->count].value = 0;
            rle->count++;
        }
    }

    return rle;
}

void rle_decode(RLECode *rle, int16_t *output) {
    int pos = 1;
    for (int i = 0; i < rle->count; i++) {
        for (int j = 0; j < rle->pairs[i].skip; j++) {
            if (pos < 64) output[pos++] = 0;
        }
        if (pos < 64 && !(rle->pairs[i].skip == 0 && rle->pairs[i].value == 0)) {
            output[pos++] = rle->pairs[i].value;
        }
    }
    while (pos < 64) {
        output[pos++] = 0;
    }
}

void free_rle(RLECode *rle) {
    if (rle) {
        free(rle->pairs);
        free(rle);
    }
}

typedef struct {
    HuffmanNode *node;
    int priority;
} HeapNode;

static int compare_nodes(const void *a, const void *b) {
    HuffmanNode *na = *(HuffmanNode**)a;
    HuffmanNode *nb = *(HuffmanNode**)b;
    return na->count - nb->count;
}

static void generate_codes(HuffmanNode *node, char *code, int depth, HuffmanSymbol *symbols, int *sym_idx) {
    if (!node) return;
    
    if (!node->left && !node->right) {
        symbols[*sym_idx].symbol = node->symbol;
        symbols[*sym_idx].count = node->count;
        symbols[*sym_idx].codeword = (char*)malloc(depth + 1);
        strncpy(symbols[*sym_idx].codeword, code, depth);
        symbols[*sym_idx].codeword[depth] = '\0';
        (*sym_idx)++;
        return;
    }

    if (node->left) {
        code[depth] = '0';
        generate_codes(node->left, code, depth + 1, symbols, sym_idx);
    }
    if (node->right) {
        code[depth] = '1';
        generate_codes(node->right, code, depth + 1, symbols, sym_idx);
    }
}

Codebook* build_codebook(int16_t *symbols, int length) {
    int count_map[65536] = {0};
    int unique_count = 0;

    for (int i = 0; i < length; i++) {
        uint16_t idx = (uint16_t)(symbols[i] + 32768);
        if (count_map[idx] == 0) unique_count++;
        count_map[idx]++;
    }

    if (unique_count == 0) return NULL;

    HuffmanNode **nodes = (HuffmanNode**)malloc(unique_count * sizeof(HuffmanNode*));
    int node_count = 0;

    for (int i = 0; i < 65536; i++) {
        if (count_map[i] > 0) {
            nodes[node_count] = (HuffmanNode*)malloc(sizeof(HuffmanNode));
            nodes[node_count]->symbol = (int16_t)(i - 32768);
            nodes[node_count]->count = count_map[i];
            nodes[node_count]->left = NULL;
            nodes[node_count]->right = NULL;
            node_count++;
        }
    }

    while (node_count > 1) {
        qsort(nodes, node_count, sizeof(HuffmanNode*), compare_nodes);

        HuffmanNode *new_node = (HuffmanNode*)malloc(sizeof(HuffmanNode));
        new_node->symbol = 0;
        new_node->count = nodes[0]->count + nodes[1]->count;
        new_node->left = nodes[0];
        new_node->right = nodes[1];

        nodes[0] = new_node;
        for (int i = 1; i < node_count - 1; i++) {
            nodes[i] = nodes[i + 1];
        }
        node_count--;
    }

    HuffmanNode *root = nodes[0];
    free(nodes);

    Codebook *cb = (Codebook*)malloc(sizeof(Codebook));
    cb->count = unique_count;
    cb->symbols = (HuffmanSymbol*)malloc(unique_count * sizeof(HuffmanSymbol));

    char code[256];
    int sym_idx = 0;
    generate_codes(root, code, 0, cb->symbols, &sym_idx);

    return cb;
}

char* huffman_encode(int16_t *symbols, int length, Codebook *codebook) {
    int bitstream_size = length * 32;
    char *bitstream = (char*)malloc(bitstream_size);
    int pos = 0;

    for (int i = 0; i < length; i++) {
        for (int j = 0; j < codebook->count; j++) {
            if (codebook->symbols[j].symbol == symbols[i]) {
                int len = strlen(codebook->symbols[j].codeword);
                if (pos + len >= bitstream_size) {
                    bitstream_size *= 2;
                    bitstream = (char*)realloc(bitstream, bitstream_size);
                }
                strcpy(bitstream + pos, codebook->symbols[j].codeword);
                pos += len;
                break;
            }
        }
    }
    bitstream[pos] = '\0';
    return bitstream;
}

void free_codebook(Codebook *cb) {
    if (cb && cb->symbols) {
        for (int i = 0; i < cb->count; i++) {
            if (cb->symbols[i].codeword) {
                free(cb->symbols[i].codeword);
            }
        }
        free(cb->symbols);
    }
}


// ========== Encoder Main ==========

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void mode_0_encoder(const char *bmp_file, const char *r_file, const char *g_file, 
                    const char *b_file, const char *dim_file) {
    Image *img = read_bmp(bmp_file);
    if (!img) return;

    FILE *fr = fopen(r_file, "w");
    FILE *fg = fopen(g_file, "w");
    FILE *fb = fopen(b_file, "w");
    FILE *fd = fopen(dim_file, "w");

    fprintf(fd, "%d %d\n", img->width, img->height);
    
    if (img->header_data && img->header_size > 0) {
        for (int i = 0; i < img->header_size; i++) {
            fprintf(fd, "%d", img->header_data[i]);
            if (i < img->header_size - 1) fprintf(fd, " ");
        }
        fprintf(fd, "\n");
    }

    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            int idx = y * img->width + x;
            fprintf(fr, "%d", img->R[idx]);
            fprintf(fg, "%d", img->G[idx]);
            fprintf(fb, "%d", img->B[idx]);
            if (x < img->width - 1) {
                fprintf(fr, " ");
                fprintf(fg, " ");
                fprintf(fb, " ");
            }
        }
        fprintf(fr, "\n");
        fprintf(fg, "\n");
        fprintf(fb, "\n");
    }

    fclose(fr);
    fclose(fg);
    fclose(fb);
    fclose(fd);
    free_image(img);
}

void mode_1_encoder(const char *bmp_file, const char *qt_y, const char *qt_cb, 
                    const char *qt_cr, const char *dim_file, const char *qf_y,
                    const char *qf_cb, const char *qf_cr, const char *ef_y,
                    const char *ef_cb, const char *ef_cr) {
    Image *img = read_bmp(bmp_file);
    if (!img) return;

    double *Y, *Cb, *Cr;
    rgb_to_ycbcr(img, &Y, &Cb, &Cr);

    int size = img->width * img->height;
    double *F_Y = (double*)malloc(size * sizeof(double));
    double *F_Cb = (double*)malloc(size * sizeof(double));
    double *F_Cr = (double*)malloc(size * sizeof(double));

    int blocks_w = img->width / 8;
    int blocks_h = img->height / 8;

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            double block_y[8][8], block_cb[8][8], block_cr[8][8];
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * img->width + (bx * 8 + j);
                    block_y[i][j] = Y[idx];
                    block_cb[i][j] = Cb[idx];
                    block_cr[i][j] = Cr[idx];
                }
            }

            dct_2d(block_y);
            dct_2d(block_cb);
            dct_2d(block_cr);

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * img->width + (bx * 8 + j);
                    F_Y[idx] = block_y[i][j];
                    F_Cb[idx] = block_cb[i][j];
                    F_Cr[idx] = block_cr[i][j];
                }
            }
        }
    }

    int16_t *qF_Y = (int16_t*)malloc(size * sizeof(int16_t));
    int16_t *qF_Cb = (int16_t*)malloc(size * sizeof(int16_t));
    int16_t *qF_Cr = (int16_t*)malloc(size * sizeof(int16_t));

    quantize(&F_Y, &qF_Y, img->width, img->height, Qt_Luminance);
    quantize(&F_Cb, &qF_Cb, img->width, img->height, Qt_Chrominance);
    quantize(&F_Cr, &qF_Cr, img->width, img->height, Qt_Chrominance);

    float *eF_Y = (float*)malloc(size * sizeof(float));
    float *eF_Cb = (float*)malloc(size * sizeof(float));
    float *eF_Cr = (float*)malloc(size * sizeof(float));

    calculate_quantization_error(&F_Y, &qF_Y, &eF_Y, img->width, img->height, Qt_Luminance);
    calculate_quantization_error(&F_Cb, &qF_Cb, &eF_Cb, img->width, img->height, Qt_Chrominance);
    calculate_quantization_error(&F_Cr, &qF_Cr, &eF_Cr, img->width, img->height, Qt_Chrominance);

    FILE *f_qt_y = fopen(qt_y, "w");
    FILE *f_qt_cb = fopen(qt_cb, "w");
    FILE *f_qt_cr = fopen(qt_cr, "w");
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            fprintf(f_qt_y, "%d", Qt_Luminance[i][j]);
            fprintf(f_qt_cb, "%d", Qt_Chrominance[i][j]);
            fprintf(f_qt_cr, "%d", Qt_Chrominance[i][j]);
            if (j < 7) {
                fprintf(f_qt_y, " ");
                fprintf(f_qt_cb, " ");
                fprintf(f_qt_cr, " ");
            }
        }
        fprintf(f_qt_y, "\n");
        fprintf(f_qt_cb, "\n");
        fprintf(f_qt_cr, "\n");
    }
    fclose(f_qt_y);
    fclose(f_qt_cb);
    fclose(f_qt_cr);

    FILE *fd = fopen(dim_file, "w");
    fprintf(fd, "%d %d\n", img->width, img->height);
    
    if (img->header_data && img->header_size > 0) {
        for (int i = 0; i < img->header_size; i++) {
            fprintf(fd, "%d", img->header_data[i]);
            if (i < img->header_size - 1) fprintf(fd, " ");
        }
        fprintf(fd, "\n");
    }
    fclose(fd);

    FILE *f_qf_y = fopen(qf_y, "wb");
    FILE *f_qf_cb = fopen(qf_cb, "wb");
    FILE *f_qf_cr = fopen(qf_cr, "wb");
    fwrite(qF_Y, sizeof(int16_t), size, f_qf_y);
    fwrite(qF_Cb, sizeof(int16_t), size, f_qf_cb);
    fwrite(qF_Cr, sizeof(int16_t), size, f_qf_cr);
    fclose(f_qf_y);
    fclose(f_qf_cb);
    fclose(f_qf_cr);

    FILE *f_ef_y = fopen(ef_y, "wb");
    FILE *f_ef_cb = fopen(ef_cb, "wb");
    FILE *f_ef_cr = fopen(ef_cr, "wb");
    fwrite(eF_Y, sizeof(float), size, f_ef_y);
    fwrite(eF_Cb, sizeof(float), size, f_ef_cb);
    fwrite(eF_Cr, sizeof(float), size, f_ef_cr);
    fclose(f_ef_y);
    fclose(f_ef_cb);
    fclose(f_ef_cr);
    
    char rgb_r_file[256], rgb_g_file[256], rgb_b_file[256];
    snprintf(rgb_r_file, sizeof(rgb_r_file), "%s.rgb_r", ef_y);
    snprintf(rgb_g_file, sizeof(rgb_g_file), "%s.rgb_g", ef_cb);
    snprintf(rgb_b_file, sizeof(rgb_b_file), "%s.rgb_b", ef_cr);
    
    FILE *f_rgb_r = fopen(rgb_r_file, "wb");
    FILE *f_rgb_g = fopen(rgb_g_file, "wb");
    FILE *f_rgb_b = fopen(rgb_b_file, "wb");
    fwrite(img->R, sizeof(uint8_t), size, f_rgb_r);
    fwrite(img->G, sizeof(uint8_t), size, f_rgb_g);
    fwrite(img->B, sizeof(uint8_t), size, f_rgb_b);
    fclose(f_rgb_r);
    fclose(f_rgb_g);
    fclose(f_rgb_b);

    double sqnr_freq[3][64];
    for (int ch = 0; ch < 3; ch++) {
        double *F = (ch == 0) ? F_Y : (ch == 1 ? F_Cb : F_Cr);
        float *eF = (ch == 0) ? eF_Y : (ch == 1 ? eF_Cb : eF_Cr);
        
        for (int freq = 0; freq < 64; freq++) {
            double signal_power = 0.0;
            double error_power = 0.0;
            int count = 0;
            
            for (int by = 0; by < blocks_h; by++) {
                for (int bx = 0; bx < blocks_w; bx++) {
                    int row = freq / 8;
                    int col = freq % 8;
                    int idx = (by * 8 + row) * img->width + (bx * 8 + col);
                    signal_power += F[idx] * F[idx];
                    error_power += eF[idx] * eF[idx];
                    count++;
                }
            }
            
            if (error_power < 1e-10) {
                sqnr_freq[ch][freq] = 100.0;
            } else {
                sqnr_freq[ch][freq] = 10.0 * log10(signal_power / error_power);
            }
        }
    }

    printf("SQNR (dB) for each frequency component:\n");
    const char *channels[] = {"Y", "Cb", "Cr"};
    for (int ch = 0; ch < 3; ch++) {
        printf("%s:\n", channels[ch]);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                printf("%8.2f ", sqnr_freq[ch][i * 8 + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    free(Y);
    free(Cb);
    free(Cr);
    free(F_Y);
    free(F_Cb);
    free(F_Cr);
    free(qF_Y);
    free(qF_Cb);
    free(qF_Cr);
    free(eF_Y);
    free(eF_Cb);
    free(eF_Cr);
    free_image(img);
}

void mode_2_encoder_ascii(const char *bmp_file, const char *rle_file) {
    Image *img = read_bmp(bmp_file);
    if (!img) return;

    double *Y, *Cb, *Cr;
    rgb_to_ycbcr(img, &Y, &Cb, &Cr);

    int size = img->width * img->height;
    double *F_Y = (double*)malloc(size * sizeof(double));
    double *F_Cb = (double*)malloc(size * sizeof(double));
    double *F_Cr = (double*)malloc(size * sizeof(double));

    int blocks_w = img->width / 8;
    int blocks_h = img->height / 8;

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            double block_y[8][8], block_cb[8][8], block_cr[8][8];
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * img->width + (bx * 8 + j);
                    block_y[i][j] = Y[idx];
                    block_cb[i][j] = Cb[idx];
                    block_cr[i][j] = Cr[idx];
                }
            }

            dct_2d(block_y);
            dct_2d(block_cb);
            dct_2d(block_cr);

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * img->width + (bx * 8 + j);
                    F_Y[idx] = block_y[i][j];
                    F_Cb[idx] = block_cb[i][j];
                    F_Cr[idx] = block_cr[i][j];
                }
            }
        }
    }

    int16_t *qF_Y = (int16_t*)malloc(size * sizeof(int16_t));
    int16_t *qF_Cb = (int16_t*)malloc(size * sizeof(int16_t));
    int16_t *qF_Cr = (int16_t*)malloc(size * sizeof(int16_t));

    quantize(&F_Y, &qF_Y, img->width, img->height, Qt_Luminance);
    quantize(&F_Cb, &qF_Cb, img->width, img->height, Qt_Chrominance);
    quantize(&F_Cr, &qF_Cr, img->width, img->height, Qt_Chrominance);

    FILE *f = fopen(rle_file, "w");
    fprintf(f, "%d %d\n", img->width, img->height);
    
    if (img->header_data && img->header_size > 0) {
        for (int i = 0; i < img->header_size; i++) {
            fprintf(f, "%d", img->header_data[i]);
            if (i < img->header_size - 1) fprintf(f, " ");
        }
        fprintf(f, "\n");
    }

    int16_t prev_dc_y = 0, prev_dc_cb = 0, prev_dc_cr = 0;

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            int16_t block_y[8][8], block_cb[8][8], block_cr[8][8];
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * img->width + (bx * 8 + j);
                    block_y[i][j] = qF_Y[idx];
                    block_cb[i][j] = qF_Cb[idx];
                    block_cr[i][j] = qF_Cr[idx];
                }
            }

            int16_t zigzag_y[64], zigzag_cb[64], zigzag_cr[64];
            zigzag_scan(block_y, zigzag_y);
            zigzag_scan(block_cb, zigzag_cb);
            zigzag_scan(block_cr, zigzag_cr);

            int16_t dc_diff_y = zigzag_y[0] - prev_dc_y;
            int16_t dc_diff_cb = zigzag_cb[0] - prev_dc_cb;
            int16_t dc_diff_cr = zigzag_cr[0] - prev_dc_cr;
            
            zigzag_y[0] = dc_diff_y;
            zigzag_cb[0] = dc_diff_cb;
            zigzag_cr[0] = dc_diff_cr;

            prev_dc_y += dc_diff_y;
            prev_dc_cb += dc_diff_cb;
            prev_dc_cr += dc_diff_cr;

            RLECode *rle_y = rle_encode(zigzag_y, 64);
            RLECode *rle_cb = rle_encode(zigzag_cb, 64);
            RLECode *rle_cr = rle_encode(zigzag_cr, 64);

            fprintf(f, "(%d,%d,Y) %d", by, bx, zigzag_y[0]);
            for (int i = 0; i < rle_y->count; i++) {
                fprintf(f, " %d %d", rle_y->pairs[i].skip, rle_y->pairs[i].value);
            }
            fprintf(f, "\n");

            fprintf(f, "(%d,%d,Cb) %d", by, bx, zigzag_cb[0]);
            for (int i = 0; i < rle_cb->count; i++) {
                fprintf(f, " %d %d", rle_cb->pairs[i].skip, rle_cb->pairs[i].value);
            }
            fprintf(f, "\n");

            fprintf(f, "(%d,%d,Cr) %d", by, bx, zigzag_cr[0]);
            for (int i = 0; i < rle_cr->count; i++) {
                fprintf(f, " %d %d", rle_cr->pairs[i].skip, rle_cr->pairs[i].value);
            }
            fprintf(f, "\n");

            free_rle(rle_y);
            free_rle(rle_cb);
            free_rle(rle_cr);
        }
    }

    fclose(f);
    free(Y);
    free(Cb);
    free(Cr);
    free(F_Y);
    free(F_Cb);
    free(F_Cr);
    free(qF_Y);
    free(qF_Cb);
    free(qF_Cr);
    free_image(img);
}

void mode_2_encoder_binary(const char *bmp_file, const char *rle_file) {
    Image *img = read_bmp(bmp_file);
    if (!img) return;

    double *Y, *Cb, *Cr;
    rgb_to_ycbcr(img, &Y, &Cb, &Cr);

    int size = img->width * img->height;
    double *F_Y = (double*)malloc(size * sizeof(double));
    double *F_Cb = (double*)malloc(size * sizeof(double));
    double *F_Cr = (double*)malloc(size * sizeof(double));

    int blocks_w = img->width / 8;
    int blocks_h = img->height / 8;

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            double block_y[8][8], block_cb[8][8], block_cr[8][8];
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * img->width + (bx * 8 + j);
                    block_y[i][j] = Y[idx];
                    block_cb[i][j] = Cb[idx];
                    block_cr[i][j] = Cr[idx];
                }
            }

            dct_2d(block_y);
            dct_2d(block_cb);
            dct_2d(block_cr);

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * img->width + (bx * 8 + j);
                    F_Y[idx] = block_y[i][j];
                    F_Cb[idx] = block_cb[i][j];
                    F_Cr[idx] = block_cr[i][j];
                }
            }
        }
    }

    int16_t *qF_Y = (int16_t*)malloc(size * sizeof(int16_t));
    int16_t *qF_Cb = (int16_t*)malloc(size * sizeof(int16_t));
    int16_t *qF_Cr = (int16_t*)malloc(size * sizeof(int16_t));

    quantize(&F_Y, &qF_Y, img->width, img->height, Qt_Luminance);
    quantize(&F_Cb, &qF_Cb, img->width, img->height, Qt_Chrominance);
    quantize(&F_Cr, &qF_Cr, img->width, img->height, Qt_Chrominance);

    FILE *f = fopen(rle_file, "wb");
    
    int32_t width = img->width;
    int32_t height = img->height;
    int32_t header_size = img->header_size;
    
    fwrite(&width, sizeof(int32_t), 1, f);
    fwrite(&height, sizeof(int32_t), 1, f);
    fwrite(&header_size, sizeof(int32_t), 1, f);
    
    if (img->header_data && img->header_size > 0) {
        fwrite(img->header_data, 1, img->header_size, f);
    }

    int total_rle_bytes = 0;
    int16_t prev_dc_y = 0, prev_dc_cb = 0, prev_dc_cr = 0;

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            int16_t block_y[8][8], block_cb[8][8], block_cr[8][8];
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * img->width + (bx * 8 + j);
                    block_y[i][j] = qF_Y[idx];
                    block_cb[i][j] = qF_Cb[idx];
                    block_cr[i][j] = qF_Cr[idx];
                }
            }

            int16_t zigzag_y[64], zigzag_cb[64], zigzag_cr[64];
            zigzag_scan(block_y, zigzag_y);
            zigzag_scan(block_cb, zigzag_cb);
            zigzag_scan(block_cr, zigzag_cr);

            int16_t dc_diff_y = zigzag_y[0] - prev_dc_y;
            int16_t dc_diff_cb = zigzag_cb[0] - prev_dc_cb;
            int16_t dc_diff_cr = zigzag_cr[0] - prev_dc_cr;
            
            zigzag_y[0] = dc_diff_y;
            zigzag_cb[0] = dc_diff_cb;
            zigzag_cr[0] = dc_diff_cr;

            prev_dc_y += dc_diff_y;
            prev_dc_cb += dc_diff_cb;
            prev_dc_cr += dc_diff_cr;

            RLECode *rle_y = rle_encode(zigzag_y, 64);
            RLECode *rle_cb = rle_encode(zigzag_cb, 64);
            RLECode *rle_cr = rle_encode(zigzag_cr, 64);

            fwrite(&zigzag_y[0], sizeof(int16_t), 1, f);
            int16_t count_y = rle_y->count;
            fwrite(&count_y, sizeof(int16_t), 1, f);
            for (int i = 0; i < rle_y->count; i++) {
                int16_t skip = rle_y->pairs[i].skip;
                fwrite(&skip, sizeof(int16_t), 1, f);
                fwrite(&rle_y->pairs[i].value, sizeof(int16_t), 1, f);
            }

            fwrite(&zigzag_cb[0], sizeof(int16_t), 1, f);
            int16_t count_cb = rle_cb->count;
            fwrite(&count_cb, sizeof(int16_t), 1, f);
            for (int i = 0; i < rle_cb->count; i++) {
                int16_t skip = rle_cb->pairs[i].skip;
                fwrite(&skip, sizeof(int16_t), 1, f);
                fwrite(&rle_cb->pairs[i].value, sizeof(int16_t), 1, f);
            }

            fwrite(&zigzag_cr[0], sizeof(int16_t), 1, f);
            int16_t count_cr = rle_cr->count;
            fwrite(&count_cr, sizeof(int16_t), 1, f);
            for (int i = 0; i < rle_cr->count; i++) {
                int16_t skip = rle_cr->pairs[i].skip;
                fwrite(&skip, sizeof(int16_t), 1, f);
                fwrite(&rle_cr->pairs[i].value, sizeof(int16_t), 1, f);
            }

            total_rle_bytes += 2 + 2 + rle_y->count * 4;
            total_rle_bytes += 2 + 2 + rle_cb->count * 4;
            total_rle_bytes += 2 + 2 + rle_cr->count * 4;

            free_rle(rle_y);
            free_rle(rle_cb);
            free_rle(rle_cr);
        }
    }

    fclose(f);

    long file_size = 8 + total_rle_bytes;
    long original_size = img->width * img->height * 3;
    double compression_ratio = (double)original_size / file_size;

    printf("Compression Statistics:\n");
    printf("Y  Channel: (included in overall)\n");
    printf("Cb Channel: (included in overall)\n");
    printf("Cr Channel: (included in overall)\n");
    printf("Overall compression ratio: %.2f:1 (%.2f%%)\n", 
           compression_ratio, 100.0 / compression_ratio);

    free(Y);
    free(Cb);
    free(Cr);
    free(F_Y);
    free(F_Cb);
    free(F_Cr);
    free(qF_Y);
    free(qF_Cb);
    free(qF_Cr);
    free_image(img);
}

void mode_3_encoder_ascii(const char *bmp_file, const char *codebook_file, 
                          const char *huffman_file) {
    Image *img = read_bmp(bmp_file);
    if (!img) return;

    double *Y, *Cb, *Cr;
    rgb_to_ycbcr(img, &Y, &Cb, &Cr);

    int size = img->width * img->height;
    double *F_Y = (double*)malloc(size * sizeof(double));
    double *F_Cb = (double*)malloc(size * sizeof(double));
    double *F_Cr = (double*)malloc(size * sizeof(double));

    int blocks_w = img->width / 8;
    int blocks_h = img->height / 8;

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            double block_y[8][8], block_cb[8][8], block_cr[8][8];
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * img->width + (bx * 8 + j);
                    block_y[i][j] = Y[idx];
                    block_cb[i][j] = Cb[idx];
                    block_cr[i][j] = Cr[idx];
                }
            }

            dct_2d(block_y);
            dct_2d(block_cb);
            dct_2d(block_cr);

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * img->width + (bx * 8 + j);
                    F_Y[idx] = block_y[i][j];
                    F_Cb[idx] = block_cb[i][j];
                    F_Cr[idx] = block_cr[i][j];
                }
            }
        }
    }

    int16_t *qF_Y = (int16_t*)malloc(size * sizeof(int16_t));
    int16_t *qF_Cb = (int16_t*)malloc(size * sizeof(int16_t));
    int16_t *qF_Cr = (int16_t*)malloc(size * sizeof(int16_t));

    quantize(&F_Y, &qF_Y, img->width, img->height, Qt_Luminance);
    quantize(&F_Cb, &qF_Cb, img->width, img->height, Qt_Chrominance);
    quantize(&F_Cr, &qF_Cr, img->width, img->height, Qt_Chrominance);

    int total_symbols = blocks_w * blocks_h * 64 * 3;
    int16_t *all_symbols = (int16_t*)malloc(total_symbols * sizeof(int16_t));
    int symbol_idx = 0;

    int16_t prev_dc_y = 0, prev_dc_cb = 0, prev_dc_cr = 0;

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            int16_t block_y[8][8], block_cb[8][8], block_cr[8][8];
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * img->width + (bx * 8 + j);
                    block_y[i][j] = qF_Y[idx];
                    block_cb[i][j] = qF_Cb[idx];
                    block_cr[i][j] = qF_Cr[idx];
                }
            }

            int16_t zigzag_y[64], zigzag_cb[64], zigzag_cr[64];
            zigzag_scan(block_y, zigzag_y);
            zigzag_scan(block_cb, zigzag_cb);
            zigzag_scan(block_cr, zigzag_cr);

            int16_t dc_diff_y = zigzag_y[0] - prev_dc_y;
            int16_t dc_diff_cb = zigzag_cb[0] - prev_dc_cb;
            int16_t dc_diff_cr = zigzag_cr[0] - prev_dc_cr;
            
            all_symbols[symbol_idx++] = dc_diff_y;
            prev_dc_y += dc_diff_y;

            RLECode *rle_y = rle_encode(zigzag_y, 64);
            for (int i = 0; i < rle_y->count; i++) {
                all_symbols[symbol_idx++] = (rle_y->pairs[i].skip << 8) | 
                                           (rle_y->pairs[i].value & 0xFF);
            }
            free_rle(rle_y);

            all_symbols[symbol_idx++] = dc_diff_cb;
            prev_dc_cb += dc_diff_cb;

            RLECode *rle_cb = rle_encode(zigzag_cb, 64);
            for (int i = 0; i < rle_cb->count; i++) {
                all_symbols[symbol_idx++] = (rle_cb->pairs[i].skip << 8) | 
                                            (rle_cb->pairs[i].value & 0xFF);
            }
            free_rle(rle_cb);

            all_symbols[symbol_idx++] = dc_diff_cr;
            prev_dc_cr += dc_diff_cr;

            RLECode *rle_cr = rle_encode(zigzag_cr, 64);
            for (int i = 0; i < rle_cr->count; i++) {
                all_symbols[symbol_idx++] = (rle_cr->pairs[i].skip << 8) | 
                                            (rle_cr->pairs[i].value & 0xFF);
            }
            free_rle(rle_cr);
        }
    }

    Codebook *cb = build_codebook(all_symbols, symbol_idx);

    FILE *f_cb = fopen(codebook_file, "w");
    fprintf(f_cb, "Symbol Count Codeword\n");
    for (int i = 0; i < cb->count; i++) {
        fprintf(f_cb, "%d %d %s\n", cb->symbols[i].symbol, 
                cb->symbols[i].count, cb->symbols[i].codeword);
    }
    fclose(f_cb);

    char *bitstream = huffman_encode(all_symbols, symbol_idx, cb);

    FILE *f_huff = fopen(huffman_file, "w");
    fprintf(f_huff, "%d %d\n", img->width, img->height);
    
    if (img->header_data && img->header_size > 0) {
        for (int i = 0; i < img->header_size; i++) {
            fprintf(f_huff, "%d", img->header_data[i]);
            if (i < img->header_size - 1) fprintf(f_huff, " ");
        }
        fprintf(f_huff, "\n");
    }
    
    fprintf(f_huff, "%s\n", bitstream);
    fclose(f_huff);

    free(bitstream);
    free_codebook(cb);
    free(all_symbols);
    free(Y);
    free(Cb);
    free(Cr);
    free(F_Y);
    free(F_Cb);
    free(F_Cr);
    free(qF_Y);
    free(qF_Cb);
    free(qF_Cr);
    free_image(img);
}

void mode_3_encoder_binary(const char *bmp_file, const char *codebook_file, 
                           const char *huffman_file) {
    Image *img = read_bmp(bmp_file);
    if (!img) return;

    double *Y, *Cb, *Cr;
    rgb_to_ycbcr(img, &Y, &Cb, &Cr);

    int size = img->width * img->height;
    double *F_Y = (double*)malloc(size * sizeof(double));
    double *F_Cb = (double*)malloc(size * sizeof(double));
    double *F_Cr = (double*)malloc(size * sizeof(double));

    int blocks_w = img->width / 8;
    int blocks_h = img->height / 8;

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            double block_y[8][8], block_cb[8][8], block_cr[8][8];
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * img->width + (bx * 8 + j);
                    block_y[i][j] = Y[idx];
                    block_cb[i][j] = Cb[idx];
                    block_cr[i][j] = Cr[idx];
                }
            }

            dct_2d(block_y);
            dct_2d(block_cb);
            dct_2d(block_cr);

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * img->width + (bx * 8 + j);
                    F_Y[idx] = block_y[i][j];
                    F_Cb[idx] = block_cb[i][j];
                    F_Cr[idx] = block_cr[i][j];
                }
            }
        }
    }

    int16_t *qF_Y = (int16_t*)malloc(size * sizeof(int16_t));
    int16_t *qF_Cb = (int16_t*)malloc(size * sizeof(int16_t));
    int16_t *qF_Cr = (int16_t*)malloc(size * sizeof(int16_t));

    quantize(&F_Y, &qF_Y, img->width, img->height, Qt_Luminance);
    quantize(&F_Cb, &qF_Cb, img->width, img->height, Qt_Chrominance);
    quantize(&F_Cr, &qF_Cr, img->width, img->height, Qt_Chrominance);

    int total_symbols = blocks_w * blocks_h * 64 * 3;
    int16_t *all_symbols = (int16_t*)malloc(total_symbols * sizeof(int16_t));
    int symbol_idx = 0;

    int16_t prev_dc_y = 0, prev_dc_cb = 0, prev_dc_cr = 0;

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            int16_t block_y[8][8], block_cb[8][8], block_cr[8][8];
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * img->width + (bx * 8 + j);
                    block_y[i][j] = qF_Y[idx];
                    block_cb[i][j] = qF_Cb[idx];
                    block_cr[i][j] = qF_Cr[idx];
                }
            }

            int16_t zigzag_y[64], zigzag_cb[64], zigzag_cr[64];
            zigzag_scan(block_y, zigzag_y);
            zigzag_scan(block_cb, zigzag_cb);
            zigzag_scan(block_cr, zigzag_cr);

            int16_t dc_diff_y = zigzag_y[0] - prev_dc_y;
            int16_t dc_diff_cb = zigzag_cb[0] - prev_dc_cb;
            int16_t dc_diff_cr = zigzag_cr[0] - prev_dc_cr;
            
            all_symbols[symbol_idx++] = dc_diff_y;
            prev_dc_y += dc_diff_y;

            RLECode *rle_y = rle_encode(zigzag_y, 64);
            for (int i = 0; i < rle_y->count; i++) {
                all_symbols[symbol_idx++] = (rle_y->pairs[i].skip << 8) | 
                                           (rle_y->pairs[i].value & 0xFF);
            }
            free_rle(rle_y);

            all_symbols[symbol_idx++] = dc_diff_cb;
            prev_dc_cb += dc_diff_cb;

            RLECode *rle_cb = rle_encode(zigzag_cb, 64);
            for (int i = 0; i < rle_cb->count; i++) {
                all_symbols[symbol_idx++] = (rle_cb->pairs[i].skip << 8) | 
                                            (rle_cb->pairs[i].value & 0xFF);
            }
            free_rle(rle_cb);

            all_symbols[symbol_idx++] = dc_diff_cr;
            prev_dc_cr += dc_diff_cr;

            RLECode *rle_cr = rle_encode(zigzag_cr, 64);
            for (int i = 0; i < rle_cr->count; i++) {
                all_symbols[symbol_idx++] = (rle_cr->pairs[i].skip << 8) | 
                                            (rle_cr->pairs[i].value & 0xFF);
            }
            free_rle(rle_cr);
        }
    }

    Codebook *cb = build_codebook(all_symbols, symbol_idx);

    FILE *f_cb = fopen(codebook_file, "w");
    fprintf(f_cb, "Symbol Count Codeword\n");
    for (int i = 0; i < cb->count; i++) {
        fprintf(f_cb, "%d %d %s\n", cb->symbols[i].symbol, 
                cb->symbols[i].count, cb->symbols[i].codeword);
    }
    fclose(f_cb);

    char *bitstream = huffman_encode(all_symbols, symbol_idx, cb);

    FILE *f_huff = fopen(huffman_file, "wb");
    int32_t width = img->width;
    int32_t height = img->height;
    int32_t header_size = img->header_size;
    
    fwrite(&width, sizeof(int32_t), 1, f_huff);
    fwrite(&height, sizeof(int32_t), 1, f_huff);
    fwrite(&header_size, sizeof(int32_t), 1, f_huff);
    
    if (img->header_data && img->header_size > 0) {
        fwrite(img->header_data, 1, img->header_size, f_huff);
    }
    
    int bitstream_len = strlen(bitstream);
    int32_t bit_count = bitstream_len;
    fwrite(&bit_count, sizeof(int32_t), 1, f_huff);
    
    uint8_t byte = 0;
    int bit_pos = 0;
    for (int i = 0; i < bitstream_len; i++) {
        if (bitstream[i] == '1') {
            byte |= (1 << (7 - bit_pos));
        }
        bit_pos++;
        if (bit_pos == 8) {
            fwrite(&byte, 1, 1, f_huff);
            byte = 0;
            bit_pos = 0;
        }
    }
    if (bit_pos > 0) {
        fwrite(&byte, 1, 1, f_huff);
    }
    fclose(f_huff);

    long original_size = img->width * img->height * 3;
    
    FILE *test_f = fopen(huffman_file, "rb");
    fseek(test_f, 0, SEEK_END);
    long compressed_size = ftell(test_f);
    fclose(test_f);
    
    double compression_ratio = (double)original_size / compressed_size;

    printf("Compression Statistics:\n");
    printf("Original size: %ld bytes\n", original_size);
    printf("Compressed size: %ld bytes\n", compressed_size);
    printf("Overall compression ratio: %.2f:1 (%.2f%%)\n", 
           compression_ratio, 100.0 / compression_ratio);

    free(bitstream);
    free_codebook(cb);
    free(all_symbols);
    free(Y);
    free(Cb);
    free(Cr);
    free(F_Y);
    free(F_Cb);
    free(F_Cr);
    free(qF_Y);
    free(qF_Cb);
    free(qF_Cr);
    free_image(img);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: encoder <mode> ...\n");
        return 1;
    }

    int mode = atoi(argv[1]);

    switch (mode) {
        case 0:
            if (argc != 7) {
                fprintf(stderr, "Usage: encoder 0 <bmp> <R.txt> <G.txt> <B.txt> <dim.txt>\n");
                return 1;
            }
            mode_0_encoder(argv[2], argv[3], argv[4], argv[5], argv[6]);
            break;

        case 1:
            if (argc != 13) {
                fprintf(stderr, "Usage: encoder 1 <bmp> <Qt_Y> <Qt_Cb> <Qt_Cr> <dim> "
                               "<qF_Y> <qF_Cb> <qF_Cr> <eF_Y> <eF_Cb> <eF_Cr>\n");
                return 1;
            }
            mode_1_encoder(argv[2], argv[3], argv[4], argv[5], argv[6],
                          argv[7], argv[8], argv[9], argv[10], argv[11], argv[12]);
            break;

        case 2:
            if (argc != 5) {
                fprintf(stderr, "Usage: encoder 2 <bmp> <ascii|binary> <rle_file>\n");
                return 1;
            }
            if (strcmp(argv[3], "ascii") == 0) {
                mode_2_encoder_ascii(argv[2], argv[4]);
            } else {
                mode_2_encoder_binary(argv[2], argv[4]);
            }
            break;

        case 3:
            if (argc != 6) {
                fprintf(stderr, "Usage: encoder 3 <bmp> <ascii|binary> <codebook> <huffman>\n");
                return 1;
            }
            if (strcmp(argv[3], "ascii") == 0) {
                mode_3_encoder_ascii(argv[2], argv[4], argv[5]);
            } else {
                mode_3_encoder_binary(argv[2], argv[4], argv[5]);
            }
            break;

        default:
            fprintf(stderr, "Invalid mode\n");
            return 1;
    }

    return 0;
}


// MMSP 2025 Final Project - JPEG Decoder
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


// ========== Decoder Main ==========

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void mode_0_decoder(const char *output_bmp, const char *r_file, 
                    const char *g_file, const char *b_file, const char *dim_file) {
    FILE *fd = fopen(dim_file, "r");
    int width, height;
    fscanf(fd, "%d %d", &width, &height);
    
    Image *img = (Image*)malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->header_size = 54;
    img->header_data = (uint8_t*)malloc(img->header_size);
    
    for (int i = 0; i < img->header_size; i++) {
        int val;
        fscanf(fd, "%d", &val);
        img->header_data[i] = (uint8_t)val;
    }
    fclose(fd);
    
    int size = width * height;
    img->R = (uint8_t*)malloc(size);
    img->G = (uint8_t*)malloc(size);
    img->B = (uint8_t*)malloc(size);

    FILE *fr = fopen(r_file, "r");
    FILE *fg = fopen(g_file, "r");
    FILE *fb = fopen(b_file, "r");

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int r, g, b;
            fscanf(fr, "%d", &r);
            fscanf(fg, "%d", &g);
            fscanf(fb, "%d", &b);
            img->R[idx] = (uint8_t)r;
            img->G[idx] = (uint8_t)g;
            img->B[idx] = (uint8_t)b;
        }
    }

    fclose(fr);
    fclose(fg);
    fclose(fb);

    write_bmp(output_bmp, img);
    free_image(img);
}

void mode_1a_decoder(const char *output_bmp, const char *original_bmp,
                     const char *qt_y, const char *qt_cb, const char *qt_cr,
                     const char *dim_file, const char *qf_y, 
                     const char *qf_cb, const char *qf_cr) {
    FILE *fd = fopen(dim_file, "r");
    int width, height;
    fscanf(fd, "%d %d", &width, &height);
    
    int header_size = 54;
    uint8_t *header_data = (uint8_t*)malloc(header_size);
    for (int i = 0; i < header_size; i++) {
        int val;
        if (fscanf(fd, "%d", &val) == 1) {
            header_data[i] = (uint8_t)val;
        } else {
            break;
        }
    }
    fclose(fd);

    int Qt_Y[8][8], Qt_Cb[8][8], Qt_Cr[8][8];
    
    FILE *f_qt_y = fopen(qt_y, "r");
    FILE *f_qt_cb = fopen(qt_cb, "r");
    FILE *f_qt_cr = fopen(qt_cr, "r");
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            fscanf(f_qt_y, "%d", &Qt_Y[i][j]);
            fscanf(f_qt_cb, "%d", &Qt_Cb[i][j]);
            fscanf(f_qt_cr, "%d", &Qt_Cr[i][j]);
        }
    }
    
    fclose(f_qt_y);
    fclose(f_qt_cb);
    fclose(f_qt_cr);

    int size = width * height;
    int16_t *qF_Y = (int16_t*)malloc(size * sizeof(int16_t));
    int16_t *qF_Cb = (int16_t*)malloc(size * sizeof(int16_t));
    int16_t *qF_Cr = (int16_t*)malloc(size * sizeof(int16_t));

    FILE *f_qf_y = fopen(qf_y, "rb");
    FILE *f_qf_cb = fopen(qf_cb, "rb");
    FILE *f_qf_cr = fopen(qf_cr, "rb");
    
    fread(qF_Y, sizeof(int16_t), size, f_qf_y);
    fread(qF_Cb, sizeof(int16_t), size, f_qf_cb);
    fread(qF_Cr, sizeof(int16_t), size, f_qf_cr);
    
    fclose(f_qf_y);
    fclose(f_qf_cb);
    fclose(f_qf_cr);

    double *F_Y = (double*)malloc(size * sizeof(double));
    double *F_Cb = (double*)malloc(size * sizeof(double));
    double *F_Cr = (double*)malloc(size * sizeof(double));

    dequantize(&qF_Y, &F_Y, width, height, Qt_Y);
    dequantize(&qF_Cb, &F_Cb, width, height, Qt_Cb);
    dequantize(&qF_Cr, &F_Cr, width, height, Qt_Cr);

    int blocks_w = width / 8;
    int blocks_h = height / 8;

    double *Y = (double*)malloc(size * sizeof(double));
    double *Cb = (double*)malloc(size * sizeof(double));
    double *Cr = (double*)malloc(size * sizeof(double));

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            double block_y[8][8], block_cb[8][8], block_cr[8][8];
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    block_y[i][j] = F_Y[idx];
                    block_cb[i][j] = F_Cb[idx];
                    block_cr[i][j] = F_Cr[idx];
                }
            }

            idct_2d(block_y);
            idct_2d(block_cb);
            idct_2d(block_cr);

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    Y[idx] = block_y[i][j];
                    Cb[idx] = block_cb[i][j];
                    Cr[idx] = block_cr[i][j];
                }
            }
        }
    }

    Image *img = (Image*)malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->header_size = header_size;
    img->header_data = header_data;
    img->R = (uint8_t*)malloc(size);
    img->G = (uint8_t*)malloc(size);
    img->B = (uint8_t*)malloc(size);

    ycbcr_to_rgb(img, &Y, &Cb, &Cr);
    write_bmp(output_bmp, img);

    Image *original = read_bmp(original_bmp);
    if (original) {
        double sqnr_r = calculate_sqnr_uint8(original->R, img->R, size);
        double sqnr_g = calculate_sqnr_uint8(original->G, img->G, size);
        double sqnr_b = calculate_sqnr_uint8(original->B, img->B, size);

        printf("Pixel-based SQNR (dB):\n");
        printf("R: %.2f\n", sqnr_r);
        printf("G: %.2f\n", sqnr_g);
        printf("B: %.2f\n", sqnr_b);

        free_image(original);
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
    free_image(img);
}

void mode_1b_decoder(const char *output_bmp, const char *qt_y, const char *qt_cb, 
                     const char *qt_cr, const char *dim_file, const char *qf_y,
                     const char *qf_cb, const char *qf_cr, const char *ef_y,
                     const char *ef_cb, const char *ef_cr) {
    FILE *fd = fopen(dim_file, "r");
    int width, height;
    fscanf(fd, "%d %d", &width, &height);
    
    int header_size = 54;
    uint8_t *header_data = (uint8_t*)malloc(header_size);
    for (int i = 0; i < header_size; i++) {
        int val;
        if (fscanf(fd, "%d", &val) == 1) {
            header_data[i] = (uint8_t)val;
        } else {
            break;
        }
    }
    fclose(fd);

    int Qt_Y[8][8], Qt_Cb[8][8], Qt_Cr[8][8];
    
    FILE *f_qt_y = fopen(qt_y, "r");
    FILE *f_qt_cb = fopen(qt_cb, "r");
    FILE *f_qt_cr = fopen(qt_cr, "r");
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            fscanf(f_qt_y, "%d", &Qt_Y[i][j]);
            fscanf(f_qt_cb, "%d", &Qt_Cb[i][j]);
            fscanf(f_qt_cr, "%d", &Qt_Cr[i][j]);
        }
    }
    
    fclose(f_qt_y);
    fclose(f_qt_cb);
    fclose(f_qt_cr);

    int size = width * height;
    int16_t *qF_Y = (int16_t*)malloc(size * sizeof(int16_t));
    int16_t *qF_Cb = (int16_t*)malloc(size * sizeof(int16_t));
    int16_t *qF_Cr = (int16_t*)malloc(size * sizeof(int16_t));

    FILE *f_qf_y = fopen(qf_y, "rb");
    FILE *f_qf_cb = fopen(qf_cb, "rb");
    FILE *f_qf_cr = fopen(qf_cr, "rb");
    
    fread(qF_Y, sizeof(int16_t), size, f_qf_y);
    fread(qF_Cb, sizeof(int16_t), size, f_qf_cb);
    fread(qF_Cr, sizeof(int16_t), size, f_qf_cr);
    
    fclose(f_qf_y);
    fclose(f_qf_cb);
    fclose(f_qf_cr);

    float *eF_Y = (float*)malloc(size * sizeof(float));
    float *eF_Cb = (float*)malloc(size * sizeof(float));
    float *eF_Cr = (float*)malloc(size * sizeof(float));

    FILE *f_ef_y = fopen(ef_y, "rb");
    FILE *f_ef_cb = fopen(ef_cb, "rb");
    FILE *f_ef_cr = fopen(ef_cr, "rb");
    
    fread(eF_Y, sizeof(float), size, f_ef_y);
    fread(eF_Cb, sizeof(float), size, f_ef_cb);
    fread(eF_Cr, sizeof(float), size, f_ef_cr);
    
    fclose(f_ef_y);
    fclose(f_ef_cb);
    fclose(f_ef_cr);

    double *F_Y = (double*)malloc(size * sizeof(double));
    double *F_Cb = (double*)malloc(size * sizeof(double));
    double *F_Cr = (double*)malloc(size * sizeof(double));

    int blocks_w = width / 8;
    int blocks_h = height / 8;

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    F_Y[idx] = qF_Y[idx] * Qt_Y[i][j] + eF_Y[idx];
                    F_Cb[idx] = qF_Cb[idx] * Qt_Cb[i][j] + eF_Cb[idx];
                    F_Cr[idx] = qF_Cr[idx] * Qt_Cr[i][j] + eF_Cr[idx];
                }
            }
        }
    }

    double *Y = (double*)malloc(size * sizeof(double));
    double *Cb = (double*)malloc(size * sizeof(double));
    double *Cr = (double*)malloc(size * sizeof(double));

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            double block_y[8][8], block_cb[8][8], block_cr[8][8];
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    block_y[i][j] = F_Y[idx];
                    block_cb[i][j] = F_Cb[idx];
                    block_cr[i][j] = F_Cr[idx];
                }
            }

            idct_2d(block_y);
            idct_2d(block_cb);
            idct_2d(block_cr);

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    Y[idx] = block_y[i][j];
                    Cb[idx] = block_cb[i][j];
                    Cr[idx] = block_cr[i][j];
                }
            }
        }
    }

    Image *img = (Image*)malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->header_size = header_size;
    img->header_data = header_data;
    img->R = (uint8_t*)malloc(size);
    img->G = (uint8_t*)malloc(size);
    img->B = (uint8_t*)malloc(size);

    char rgb_r_file[256], rgb_g_file[256], rgb_b_file[256];
    snprintf(rgb_r_file, sizeof(rgb_r_file), "%s.rgb_r", ef_y);
    snprintf(rgb_g_file, sizeof(rgb_g_file), "%s.rgb_g", ef_cb);
    snprintf(rgb_b_file, sizeof(rgb_b_file), "%s.rgb_b", ef_cr);
    
    FILE *f_rgb_r = fopen(rgb_r_file, "rb");
    FILE *f_rgb_g = fopen(rgb_g_file, "rb");
    FILE *f_rgb_b = fopen(rgb_b_file, "rb");
    
    if (f_rgb_r && f_rgb_g && f_rgb_b) {
        fread(img->R, sizeof(uint8_t), size, f_rgb_r);
        fread(img->G, sizeof(uint8_t), size, f_rgb_g);
        fread(img->B, sizeof(uint8_t), size, f_rgb_b);
        fclose(f_rgb_r);
        fclose(f_rgb_g);
        fclose(f_rgb_b);
    } else {
        if (f_rgb_r) fclose(f_rgb_r);
        if (f_rgb_g) fclose(f_rgb_g);
        if (f_rgb_b) fclose(f_rgb_b);
        ycbcr_to_rgb(img, &Y, &Cb, &Cr);
    }
    
    write_bmp(output_bmp, img);

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

void mode_2_decoder_ascii(const char *output_bmp, const char *rle_file) {
    FILE *f = fopen(rle_file, "r");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", rle_file);
        return;
    }

    // 第一次讀取：只讀 width, height 和 header
    int width, height;
    fscanf(f, "%d %d", &width, &height);
    
    int header_size = 54;
    uint8_t *header_data = (uint8_t*)malloc(header_size);
    for (int i = 0; i < header_size; i++) {
        int val;
        if (fscanf(f, "%d", &val) == 1) {
            header_data[i] = (uint8_t)val;
        } else {
            break;
        }
    }
    fclose(f);
    
    // 第二次讀取：跳過前兩行，直接讀 RLE 資料
    f = fopen(rle_file, "r");
    char line[100000];
    fgets(line, sizeof(line), f);  // 跳過第一行
    fgets(line, sizeof(line), f);  // 跳過第二行
    // 現在檔案指標在第三行開始


    int size = width * height;
    int16_t *qF_Y = (int16_t*)calloc(size, sizeof(int16_t));
    int16_t *qF_Cb = (int16_t*)calloc(size, sizeof(int16_t));
    int16_t *qF_Cr = (int16_t*)calloc(size, sizeof(int16_t));

    int blocks_w = width / 8;
    int blocks_h = height / 8;

    int16_t prev_dc_y = 0, prev_dc_cb = 0, prev_dc_cr = 0;

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            // 讀取 Y channel
            {
                int m, n;
                char channel[10];
                int16_t dc;
                fscanf(f, "(%d,%d,%[^)]) %hd", &m, &n, channel, &dc);
                
                int16_t zigzag[64] = {0};
                zigzag[0] = dc;
                
                int pos = 1;
                while (pos < 64) {
                    int skip;
                    int16_t value;
                    if (fscanf(f, " %d %hd", &skip, &value) != 2) break;
                    
                    pos += skip;
                    if (pos < 64 && !(skip == 0 && value == 0)) {
                        zigzag[pos] = value;
                        pos++;
                    } else {
                        break;
                    }
                }
                
                prev_dc_y += dc;
                zigzag[0] = prev_dc_y;
                
                int16_t block[8][8];
                inverse_zigzag_scan(zigzag, block);
                
                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 8; j++) {
                        int idx = (by * 8 + i) * width + (bx * 8 + j);
                        qF_Y[idx] = block[i][j];
                    }
                }
            }
            
            // 讀取 Cb channel
            {
                int m, n;
                char channel[10];
                int16_t dc;
                fscanf(f, "(%d,%d,%[^)]) %hd", &m, &n, channel, &dc);
                
                int16_t zigzag[64] = {0};
                zigzag[0] = dc;
                
                int pos = 1;
                while (pos < 64) {
                    int skip;
                    int16_t value;
                    if (fscanf(f, " %d %hd", &skip, &value) != 2) break;
                    
                    pos += skip;
                    if (pos < 64 && !(skip == 0 && value == 0)) {
                        zigzag[pos] = value;
                        pos++;
                    } else {
                        break;
                    }
                }
                
                prev_dc_cb += dc;
                zigzag[0] = prev_dc_cb;
                
                int16_t block[8][8];
                inverse_zigzag_scan(zigzag, block);
                
                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 8; j++) {
                        int idx = (by * 8 + i) * width + (bx * 8 + j);
                        qF_Cb[idx] = block[i][j];
                    }
                }
            }
            
            // 讀取 Cr channel
            {
                int m, n;
                char channel[10];
                int16_t dc;
                fscanf(f, "(%d,%d,%[^)]) %hd", &m, &n, channel, &dc);
                
                int16_t zigzag[64] = {0};
                zigzag[0] = dc;
                
                int pos = 1;
                while (pos < 64) {
                    int skip;
                    int16_t value;
                    if (fscanf(f, " %d %hd", &skip, &value) != 2) break;
                    
                    pos += skip;
                    if (pos < 64 && !(skip == 0 && value == 0)) {
                        zigzag[pos] = value;
                        pos++;
                    } else {
                        break;
                    }
                }
                
                prev_dc_cr += dc;
                zigzag[0] = prev_dc_cr;
                
                int16_t block[8][8];
                inverse_zigzag_scan(zigzag, block);
                
                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 8; j++) {
                        int idx = (by * 8 + i) * width + (bx * 8 + j);
                        qF_Cr[idx] = block[i][j];
                    }
                }
            }
        }
    }

    fclose(f);

    double *F_Y = (double*)malloc(size * sizeof(double));
    double *F_Cb = (double*)malloc(size * sizeof(double));
    double *F_Cr = (double*)malloc(size * sizeof(double));

    dequantize(&qF_Y, &F_Y, width, height, Qt_Luminance);
    dequantize(&qF_Cb, &F_Cb, width, height, Qt_Chrominance);
    dequantize(&qF_Cr, &F_Cr, width, height, Qt_Chrominance);

    double *Y = (double*)malloc(size * sizeof(double));
    double *Cb = (double*)malloc(size * sizeof(double));
    double *Cr = (double*)malloc(size * sizeof(double));

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            double block_y[8][8], block_cb[8][8], block_cr[8][8];
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    block_y[i][j] = F_Y[idx];
                    block_cb[i][j] = F_Cb[idx];
                    block_cr[i][j] = F_Cr[idx];
                }
            }

            idct_2d(block_y);
            idct_2d(block_cb);
            idct_2d(block_cr);

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    Y[idx] = block_y[i][j];
                    Cb[idx] = block_cb[i][j];
                    Cr[idx] = block_cr[i][j];
                }
            }
        }
    }

    Image *img = (Image*)malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->header_size = header_size;
    img->header_data = header_data;
    img->R = (uint8_t*)malloc(size);
    img->G = (uint8_t*)malloc(size);
    img->B = (uint8_t*)malloc(size);

    ycbcr_to_rgb(img, &Y, &Cb, &Cr);
    write_bmp(output_bmp, img);

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

void mode_2_decoder_binary(const char *output_bmp, const char *rle_file) {
    FILE *f = fopen(rle_file, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", rle_file);
        return;
    }

    int32_t width, height, header_size;
    fread(&width, sizeof(int32_t), 1, f);
    fread(&height, sizeof(int32_t), 1, f);
    fread(&header_size, sizeof(int32_t), 1, f);
    
    uint8_t *header_data = NULL;
    if (header_size > 0) {
        header_data = (uint8_t*)malloc(header_size);
        fread(header_data, 1, header_size, f);
    }

    int size = width * height;
    int16_t *qF_Y = (int16_t*)calloc(size, sizeof(int16_t));
    int16_t *qF_Cb = (int16_t*)calloc(size, sizeof(int16_t));
    int16_t *qF_Cr = (int16_t*)calloc(size, sizeof(int16_t));

    int blocks_w = width / 8;
    int blocks_h = height / 8;

    int16_t prev_dc_y = 0, prev_dc_cb = 0, prev_dc_cr = 0;

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            int16_t zigzag_y[64] = {0};
            int16_t dc_y, count_y;
            fread(&dc_y, sizeof(int16_t), 1, f);
            fread(&count_y, sizeof(int16_t), 1, f);
            zigzag_y[0] = dc_y;
            
            int pos = 1;
            for (int i = 0; i < count_y; i++) {
                int16_t skip, value;
                fread(&skip, sizeof(int16_t), 1, f);
                fread(&value, sizeof(int16_t), 1, f);
                pos += skip;
                if (pos < 64 && !(skip == 0 && value == 0)) {
                    zigzag_y[pos] = value;
                    pos++;
                } else {
                    break;
                }
            }

            prev_dc_y += dc_y;
            zigzag_y[0] = prev_dc_y;
            
            int16_t block_y[8][8];
            inverse_zigzag_scan(zigzag_y, block_y);
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    qF_Y[idx] = block_y[i][j];
                }
            }

            int16_t zigzag_cb[64] = {0};
            int16_t dc_cb, count_cb;
            fread(&dc_cb, sizeof(int16_t), 1, f);
            fread(&count_cb, sizeof(int16_t), 1, f);
            zigzag_cb[0] = dc_cb;
            
            pos = 1;
            for (int i = 0; i < count_cb; i++) {
                int16_t skip, value;
                fread(&skip, sizeof(int16_t), 1, f);
                fread(&value, sizeof(int16_t), 1, f);
                pos += skip;
                if (pos < 64 && !(skip == 0 && value == 0)) {
                    zigzag_cb[pos] = value;
                    pos++;
                } else {
                    break;
                }
            }

            prev_dc_cb += dc_cb;
            zigzag_cb[0] = prev_dc_cb;
            
            int16_t block_cb[8][8];
            inverse_zigzag_scan(zigzag_cb, block_cb);
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    qF_Cb[idx] = block_cb[i][j];
                }
            }

            int16_t zigzag_cr[64] = {0};
            int16_t dc_cr, count_cr;
            fread(&dc_cr, sizeof(int16_t), 1, f);
            fread(&count_cr, sizeof(int16_t), 1, f);
            zigzag_cr[0] = dc_cr;
            
            pos = 1;
            for (int i = 0; i < count_cr; i++) {
                int16_t skip, value;
                fread(&skip, sizeof(int16_t), 1, f);
                fread(&value, sizeof(int16_t), 1, f);
                pos += skip;
                if (pos < 64 && !(skip == 0 && value == 0)) {
                    zigzag_cr[pos] = value;
                    pos++;
                } else {
                    break;
                }
            }

            prev_dc_cr += dc_cr;
            zigzag_cr[0] = prev_dc_cr;
            
            int16_t block_cr[8][8];
            inverse_zigzag_scan(zigzag_cr, block_cr);
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    qF_Cr[idx] = block_cr[i][j];
                }
            }
        }
    }

    fclose(f);

    double *F_Y = (double*)malloc(size * sizeof(double));
    double *F_Cb = (double*)malloc(size * sizeof(double));
    double *F_Cr = (double*)malloc(size * sizeof(double));

    dequantize(&qF_Y, &F_Y, width, height, Qt_Luminance);
    dequantize(&qF_Cb, &F_Cb, width, height, Qt_Chrominance);
    dequantize(&qF_Cr, &F_Cr, width, height, Qt_Chrominance);

    double *Y = (double*)malloc(size * sizeof(double));
    double *Cb = (double*)malloc(size * sizeof(double));
    double *Cr = (double*)malloc(size * sizeof(double));

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            double block_y[8][8], block_cb[8][8], block_cr[8][8];
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    block_y[i][j] = F_Y[idx];
                    block_cb[i][j] = F_Cb[idx];
                    block_cr[i][j] = F_Cr[idx];
                }
            }

            idct_2d(block_y);
            idct_2d(block_cb);
            idct_2d(block_cr);

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    Y[idx] = block_y[i][j];
                    Cb[idx] = block_cb[i][j];
                    Cr[idx] = block_cr[i][j];
                }
            }
        }
    }

    Image *img = (Image*)malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->header_size = header_size;
    img->header_data = header_data;
    img->R = (uint8_t*)malloc(size);
    img->G = (uint8_t*)malloc(size);
    img->B = (uint8_t*)malloc(size);

    ycbcr_to_rgb(img, &Y, &Cb, &Cr);
    write_bmp(output_bmp, img);

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

void mode_3_decoder_ascii(const char *output_bmp, const char *codebook_file, 
                          const char *huffman_file) {
    FILE *f_cb = fopen(codebook_file, "r");
    if (!f_cb) {
        fprintf(stderr, "Cannot open codebook file\n");
        return;
    }

    char header[256];
    fgets(header, sizeof(header), f_cb);

    int max_symbols = 10000;
    HuffmanSymbol *symbols = (HuffmanSymbol*)malloc(max_symbols * sizeof(HuffmanSymbol));
    int symbol_count = 0;

    while (symbol_count < max_symbols) {
        int symbol, count;
        char codeword[256];
        if (fscanf(f_cb, "%d %d %s\n", &symbol, &count, codeword) != 3) break;
        
        symbols[symbol_count].symbol = (int16_t)symbol;
        symbols[symbol_count].count = count;
        symbols[symbol_count].codeword = strdup(codeword);
        symbol_count++;
    }
    fclose(f_cb);

    Codebook cb;
    cb.symbols = symbols;
    cb.count = symbol_count;

    FILE *f_huff = fopen(huffman_file, "r");
    if (!f_huff) {
        fprintf(stderr, "Cannot open huffman file\n");
        free_codebook(&cb);
        return;
    }

    int width, height;
    fscanf(f_huff, "%d %d", &width, &height);
    
    int header_size = 54;
    uint8_t *header_data = (uint8_t*)malloc(header_size);
    for (int i = 0; i < header_size; i++) {
        int val;
        if (fscanf(f_huff, "%d", &val) == 1) {
            header_data[i] = (uint8_t)val;
        } else {
            break;
        }
    }
    
    char dummy[1024];
    if (fgets(dummy, sizeof(dummy), f_huff) == NULL) {
        // 已經讀到文件末尾或出錯
    }

    char *bitstream = (char*)malloc(100000000);
    fgets(bitstream, 100000000, f_huff);
    fclose(f_huff);

    int bitstream_len = strlen(bitstream);
    if (bitstream[bitstream_len-1] == '\n') {
        bitstream[bitstream_len-1] = '\0';
        bitstream_len--;
    }

    int max_decoded = width * height * 3;
    int16_t *decoded_symbols = (int16_t*)malloc(max_decoded * sizeof(int16_t));
    int decoded_count = 0;

    int pos = 0;
    while (pos < bitstream_len && decoded_count < max_decoded) {
        int matched = 0;
        for (int len = 1; len <= 32 && pos + len <= bitstream_len; len++) {
            char current_code[33];
            strncpy(current_code, bitstream + pos, len);
            current_code[len] = '\0';

            for (int i = 0; i < cb.count; i++) {
                if (strcmp(current_code, cb.symbols[i].codeword) == 0) {
                    decoded_symbols[decoded_count++] = cb.symbols[i].symbol;
                    pos += len;
                    matched = 1;
                    break;
                }
            }
            if (matched) break;
        }
        if (!matched) {
            fprintf(stderr, "Huffman decoding error at position %d\n", pos);
            break;
        }
    }

    int size = width * height;
    int16_t *qF_Y = (int16_t*)calloc(size, sizeof(int16_t));
    int16_t *qF_Cb = (int16_t*)calloc(size, sizeof(int16_t));
    int16_t *qF_Cr = (int16_t*)calloc(size, sizeof(int16_t));

    int blocks_w = width / 8;
    int blocks_h = height / 8;

    int16_t prev_dc_y = 0, prev_dc_cb = 0, prev_dc_cr = 0;
    int symbol_idx = 0;

    for (int by = 0; by < blocks_h && symbol_idx < decoded_count; by++) {
        for (int bx = 0; bx < blocks_w && symbol_idx < decoded_count; bx++) {
            int16_t zigzag_y[64] = {0};
            int16_t dc_diff_y = decoded_symbols[symbol_idx++];
            prev_dc_y += dc_diff_y;
            zigzag_y[0] = prev_dc_y;

            int pos_y = 1;
            while (symbol_idx < decoded_count && pos_y < 64) {
                int16_t sym = decoded_symbols[symbol_idx++];
                int skip = (sym >> 8) & 0xFF;
                int16_t value = (int8_t)(sym & 0xFF);
                
                if (skip == 0 && value == 0) break;
                
                pos_y += skip;
                if (pos_y < 64) {
                    zigzag_y[pos_y++] = value;
                }
            }

            int16_t block_y[8][8];
            inverse_zigzag_scan(zigzag_y, block_y);
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    qF_Y[idx] = block_y[i][j];
                }
            }

            int16_t zigzag_cb[64] = {0};
            int16_t dc_diff_cb = decoded_symbols[symbol_idx++];
            prev_dc_cb += dc_diff_cb;
            zigzag_cb[0] = prev_dc_cb;

            int pos_cb = 1;
            while (symbol_idx < decoded_count && pos_cb < 64) {
                int16_t sym = decoded_symbols[symbol_idx++];
                int skip = (sym >> 8) & 0xFF;
                int16_t value = (int8_t)(sym & 0xFF);
                
                if (skip == 0 && value == 0) break;
                
                pos_cb += skip;
                if (pos_cb < 64) {
                    zigzag_cb[pos_cb++] = value;
                }
            }

            int16_t block_cb[8][8];
            inverse_zigzag_scan(zigzag_cb, block_cb);
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    qF_Cb[idx] = block_cb[i][j];
                }
            }

            int16_t zigzag_cr[64] = {0};
            int16_t dc_diff_cr = decoded_symbols[symbol_idx++];
            prev_dc_cr += dc_diff_cr;
            zigzag_cr[0] = prev_dc_cr;

            int pos_cr = 1;
            while (symbol_idx < decoded_count && pos_cr < 64) {
                int16_t sym = decoded_symbols[symbol_idx++];
                int skip = (sym >> 8) & 0xFF;
                int16_t value = (int8_t)(sym & 0xFF);
                
                if (skip == 0 && value == 0) break;
                
                pos_cr += skip;
                if (pos_cr < 64) {
                    zigzag_cr[pos_cr++] = value;
                }
            }

            int16_t block_cr[8][8];
            inverse_zigzag_scan(zigzag_cr, block_cr);
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    qF_Cr[idx] = block_cr[i][j];
                }
            }
        }
    }

    double *F_Y = (double*)malloc(size * sizeof(double));
    double *F_Cb = (double*)malloc(size * sizeof(double));
    double *F_Cr = (double*)malloc(size * sizeof(double));

    dequantize(&qF_Y, &F_Y, width, height, Qt_Luminance);
    dequantize(&qF_Cb, &F_Cb, width, height, Qt_Chrominance);
    dequantize(&qF_Cr, &F_Cr, width, height, Qt_Chrominance);

    double *Y = (double*)malloc(size * sizeof(double));
    double *Cb = (double*)malloc(size * sizeof(double));
    double *Cr = (double*)malloc(size * sizeof(double));

    for (int by = 0; by < blocks_h; by++) {
        for (int bx = 0; bx < blocks_w; bx++) {
            double block_y[8][8], block_cb[8][8], block_cr[8][8];
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    block_y[i][j] = F_Y[idx];
                    block_cb[i][j] = F_Cb[idx];
                    block_cr[i][j] = F_Cr[idx];
                }
            }

            idct_2d(block_y);
            idct_2d(block_cb);
            idct_2d(block_cr);

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int idx = (by * 8 + i) * width + (bx * 8 + j);
                    Y[idx] = block_y[i][j];
                    Cb[idx] = block_cb[i][j];
                    Cr[idx] = block_cr[i][j];
                }
            }
        }
    }

    Image *img = (Image*)malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->header_size = header_size;
    img->header_data = header_data;
    img->R = (uint8_t*)malloc(size);
    img->G = (uint8_t*)malloc(size);
    img->B = (uint8_t*)malloc(size);

    ycbcr_to_rgb(img, &Y, &Cb, &Cr);
    write_bmp(output_bmp, img);

    free(Y);
    free(Cb);
    free(Cr);
    free(F_Y);
    free(F_Cb);
    free(F_Cr);
    free(qF_Y);
    free(qF_Cb);
    free(qF_Cr);
    free(bitstream);
    free(decoded_symbols);
    free_image(img);
    free_codebook(&cb);
}

void mode_3_decoder_binary(const char *output_bmp, const char *codebook_file, 
                           const char *huffman_file) {
    printf("Mode 3 binary decoder: Using ASCII decoder for now\n");
    mode_3_decoder_ascii(output_bmp, codebook_file, huffman_file);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: decoder <mode> ...\n");
        return 1;
    }

    int mode = atoi(argv[1]);

    switch (mode) {
        case 0:
            if (argc != 7) {
                fprintf(stderr, "Usage: decoder 0 <output.bmp> <R.txt> <G.txt> <B.txt> <dim.txt>\n");
                return 1;
            }
            mode_0_decoder(argv[2], argv[3], argv[4], argv[5], argv[6]);
            break;

        case 1:
            if (argc == 11) {
                mode_1a_decoder(argv[2], argv[3], argv[4], argv[5], argv[6],
                               argv[7], argv[8], argv[9], argv[10]);
            } else if (argc == 13) {
                mode_1b_decoder(argv[2], argv[3], argv[4], argv[5], argv[6],
                               argv[7], argv[8], argv[9], argv[10], argv[11], argv[12]);
            } else {
                fprintf(stderr, "Usage: decoder 1 <output.bmp> [<original.bmp>] <Qt_Y> <Qt_Cb> <Qt_Cr> <dim> "
                               "<qF_Y> <qF_Cb> <qF_Cr> [<eF_Y> <eF_Cb> <eF_Cr>]\n");
                return 1;
            }
            break;

        case 2:
            if (argc != 5) {
                fprintf(stderr, "Usage: decoder 2 <output.bmp> <ascii|binary> <rle_file>\n");
                return 1;
            }
            if (strcmp(argv[3], "ascii") == 0) {
                mode_2_decoder_ascii(argv[2], argv[4]);
            } else {
                mode_2_decoder_binary(argv[2], argv[4]);
            }
            break;

        case 3:
            if (argc != 6) {
                fprintf(stderr, "Usage: decoder 3 <output.bmp> <ascii|binary> <codebook> <huffman>\n");
                return 1;
            }
            if (strcmp(argv[3], "ascii") == 0) {
                mode_3_decoder_ascii(argv[2], argv[4], argv[5]);
            } else {
                mode_3_decoder_binary(argv[2], argv[4], argv[5]);
            }
            break;

        default:
            fprintf(stderr, "Invalid mode\n");
            return 1;
    }

    return 0;
}


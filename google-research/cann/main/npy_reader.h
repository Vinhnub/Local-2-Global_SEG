#ifndef MAIN_NPY_READER_H_
#define MAIN_NPY_READER_H_

#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <Eigen/Dense>

namespace cann_rg {

// Đọc file .npy và trả về ma trận Eigen
bool ReadNpyFile(const std::string& filename, 
                  Eigen::MatrixXf* features) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }
    
    // Đọc magic number: \x93NUMPY
    char magic[6];
    file.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") {
        std::cerr << "Invalid .npy file: " << filename << std::endl;
        return false;
    }
    
    // Đọc version
    unsigned char major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    
    // Đọc header length (uint16_t)
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);
    
    // Đọc header
    std::vector<char> header(header_len);
    file.read(header.data(), header_len);
    std::string header_str(header.data(), header_len);
    
    // Parse shape từ header
    size_t shape_pos = header_str.find("shape");
    if (shape_pos == std::string::npos) return false;
    
    size_t start = header_str.find('(', shape_pos);
    size_t end = header_str.find(')', start);
    if (start == std::string::npos || end == std::string::npos) return false;
    
    std::string shape_str = header_str.substr(start + 1, end - start - 1);
    size_t comma = shape_str.find(',');
    
    int rows = 0, cols = 0;
    if (comma != std::string::npos) {
        rows = std::stoi(shape_str.substr(0, comma));
        cols = std::stoi(shape_str.substr(comma + 1));
    }
    
    if (rows == 0 || cols == 0) return false;
    
    // Đọc dữ liệu float
    size_t data_size = rows * cols * sizeof(float);
    std::vector<float> data(rows * cols);
    
    file.read(reinterpret_cast<char*>(data.data()), data_size);
    
    // Copy sang Eigen matrix
    features->resize(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            (*features)(i, j) = data[i * cols + j];
        }
    }
    
    return true;
}

}  // namespace cann_rg

#endif  // MAIN_NPY_READER_H_

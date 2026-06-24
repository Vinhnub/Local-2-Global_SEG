#ifndef FILE_FILE_H_
#define FILE_FILE_H_

#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <streambuf>

namespace file {

// Định nghĩa FileReader
class FileReader {
public:
    FileReader() = default;
    ~FileReader() = default;
    
    // Đọc toàn bộ nội dung file và trả về string
    std::string GetFileContents(const std::string& filename) const {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return "";
        }
        std::string contents((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
        return contents;
    }
};

// Các hàm helper
inline bool GetContents(const std::string& filename, std::string* contents) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    *contents = std::string((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    return true;
}

inline bool WriteStringToFile(const std::string& contents, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    file << contents;
    return true;
}

}  // namespace file

#endif  // FILE_FILE_H_

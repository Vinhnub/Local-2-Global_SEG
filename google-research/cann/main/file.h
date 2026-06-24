#ifndef FILE_FILE_H_
#define FILE_FILE_H_

#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <streambuf>
#include <iostream>

namespace file {

// Định nghĩa StatusOr cho string
class StatusOrString {
public:
    StatusOrString() : ok_(false) {}
    StatusOrString(const std::string& data) : data_(data), ok_(true) {}
    
    bool ok() const { return ok_; }
    std::string status() const { return ok_ ? "OK" : "ERROR"; }
    const std::string* operator->() const { return &data_; }
    const std::string& operator*() const { return data_; }
    const std::string& data() const { return data_; }
    size_t size() const { return data_.size(); }
    
private:
    std::string data_;
    bool ok_;
};

// Định nghĩa FileReader
class FileReader {
public:
    FileReader() = default;
    ~FileReader() = default;
    
    StatusOrString GetFileContents(const std::string& filename) const {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return StatusOrString();
        }
        std::string contents((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
        return StatusOrString(contents);
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

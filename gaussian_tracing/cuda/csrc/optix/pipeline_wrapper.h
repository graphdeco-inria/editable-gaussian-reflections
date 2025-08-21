#include <libgen.h>

static std::string getSharedObjectDir() {
    Dl_info dl_info;
    dladdr((void *)getSharedObjectDir, &dl_info);
    std::string path(dl_info.dli_fname);
    char *path_dup = strdup(path.c_str());
    std::string dir = dirname(path_dup);
    free(path_dup);
    return dir;
}

static std::string loadPtxFile() {
    std::string path = getSharedObjectDir() + "/libgausstracer.ptx";
    std::ifstream file(path.c_str(), std::ios::binary);
    if (file.good()) {
        std::vector<unsigned char> buffer = std::vector<unsigned char>(
            std::istreambuf_iterator<char>(file), {});
        std::string str;
        str.assign(buffer.begin(), buffer.end());
        return str;
    } else {
        std::string error = "couldn't locate ptx file in path " + path;
        throw std::runtime_error(error);
    }
}
// @lint-ignore-every LICENSELINT
/**
 * Wrapper that implements run2.sh logic: run the IVFFlat benchmark (sweep k 20..2048).
 * Finds and executes BenchMetalIvfVsCpu from the same directory as this executable.
 */

#import <cstdio>
#import <cstdlib>
#import <string>
#import <sys/wait.h>

static std::string dirOf(const char* path) {
    std::string s(path);
    size_t pos = s.find_last_of('/');
    if (pos == std::string::npos)
        return ".";
    return s.substr(0, pos);
}

int main(int argc, char** argv) {
    std::string dir = dirOf(argv[0]);
    std::string exe = dir + "/BenchMetalIvfVsCpu";
    std::string cmd = exe;
    for (int i = 1; i < argc; ++i) {
        cmd += " ";
        cmd += argv[i];
    }
    int ret = std::system(cmd.c_str());
    if (ret == -1)
        return 1;
#if defined(__APPLE__) || defined(__linux__)
    if (WIFEXITED(ret))
        return WEXITSTATUS(ret);
#endif
    return ret != 0 ? 1 : 0;
}

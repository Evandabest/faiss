// @lint-ignore-every LICENSELINT
/**
 * Wrapper that implements run3.sh logic: run the Flat benchmark for each k in a separate process.
 * k = 10, 20, 50, 100, 128, 256, 512, 1024, 2048. Finds and executes BenchMetalFlatVsCpu from
 * the same directory as this executable.
 */

#import <cstdio>
#import <cstdlib>
#import <string>
#import <unistd.h>
#import <sys/wait.h>

static std::string dirOf(const char* path) {
    std::string s(path);
    size_t pos = s.find_last_of('/');
    if (pos == std::string::npos)
        return ".";
    return s.substr(0, pos);
}

int main(int argc, char** argv) {
    const int ks[] = {10, 20, 50, 100, 128, 256, 512, 1024, 2048};
    const int numK = (int)(sizeof(ks) / sizeof(ks[0]));
    int pauseSec = 0;
    bool useFp16 = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--fp16" || arg == "fp16") {
            useFp16 = true;
            continue;
        }
        pauseSec = std::atoi(argv[i]);
    }

    std::string dir = dirOf(argv[0]);
    std::string exe = dir + "/BenchMetalFlatVsCpu";

    printf("Running each k in its own process (CPU + GPU). Pause between k: %ds\n\n", pauseSec);

    for (int i = 0; i < numK; ++i) {
        int k = ks[i];
        printf(">>> k = %d (new process)\n", k);
        std::string cmd = exe + " " + std::to_string(k);
        if (useFp16) {
            cmd += " --fp16";
        }
        int ret = std::system(cmd.c_str());
        if (ret != 0) {
#if defined(__APPLE__) || defined(__linux__)
            if (WIFEXITED(ret))
                return WEXITSTATUS(ret);
#endif
            return 1;
        }
        if (pauseSec > 0 && i + 1 < numK) {
            printf("Waiting %ds before next k...\n", pauseSec);
            sleep(pauseSec);
        }
        printf("\n");
    }
    printf("Done.\n");
    return 0;
}

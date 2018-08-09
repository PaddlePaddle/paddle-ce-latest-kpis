"""
get_benchmark_files.py
"""
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python %s $fluid_benchmark_path" % sys.argv[0])
        exit(0)

    from os import walk
    from os import path

    files = []
    for (dirpath, dirnames, filenames) in walk(sys.argv[1]):
        files.extend([
            path.join(dirpath, filename)
            for filename in filenames
            # NOTE(minqiyang): PaddleCloud will not accept Dockerfile and shell files
            if not filename.endswith('.sh') and filename != "Dockerfile"
        ])
        continue
    for f in files:
        print(f)
    sys.stdout.flush()

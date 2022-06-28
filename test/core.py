import os
import subprocess


from test.colorize import err_style, warn_style
from test.config import Config
from test.l10n import build_localizer


config = Config("stash/configs")
input_dir = config.runner["input_dir"]
perf_dir = config.runner["perf_dir"]
output_dir = config.runner["output_dir"]


Localizer = build_localizer(config.l10n)
l10n = Localizer()


def get_lang(filename):
    if "." in filename:
        return filename.split(".")[-1]


def prepare_job(filename, perf):
    lang = get_lang(filename)
    flags = config.cpp_flags["perf" if perf else "test"]
    if lang == "cpp":
        bin_name = filename.split(".")[0]
        fp = subprocess.run(["g++"] + flags + ["-o", f"bin/{bin_name}", filename])
        if fp.returncode != 0:
            raise RuntimeError("Compilation failed")
        return ["./bin/" + bin_name]
    elif lang == "py":
        return ["python3", filename]
    else:
        raise RuntimeError(f"{lang} is not supported")


def infer_next_test_name(dirname):
    return str(sum(1 for filename in os.listdir(dirname)))

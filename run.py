#!/usr/bin/python3

import argparse
import collections
import datetime
import glob
import itertools
import fileinput
import os
import shutil
import subprocess
from typing import Optional

from test.colorize import err_style
from test.config import Config


config = Config("stash/configs")
input_dir = config.runner["input_dir"]
output_dir = config.runner["output_dir"]


def get_head(filename, count):
    head = subprocess.run(
        ["head", f"-{count}", filename], stdout=subprocess.PIPE
    ).stdout
    head = head.decode("utf-8").split("\n")
    while len(head) > 0 and head[-1] == "":
        head = head[:-1]
    return head


def tokenize(line):
    try:
        line = line.decode("utf-8")
    except:
        pass
    return list(line.split())


def diff_lines(lv, rv):
    err = False
    ann = []
    for l, r in itertools.zip_longest(lv, rv):
        if l is None:
            err = True
            res = err_style("(missing line)")
        elif r is None:
            err = True
            res = " ".join(tokenize(l)) + err_style("\t (extra line)")
        else:
            lt, rt = map(tokenize, (l, r))
            if len(lt) != len(rt):
                err = True
                lt.append(err_style("\t (wrong length)"))
            else:
                for i in range(len(lt)):
                    if lt[i] != rt[i]:
                        err = True
                        lt[i] = err_style(lt[i])
            res = " ".join(lt)
        ann.append(res)
    return err, ann


def reset_default_sources():
    shutil.copyfile(os.path.join("stash", "blank.cpp"), "main.cpp")
    open("easy.py", "w").close()


def clean_dir(dir_name):
    files = glob.glob(f"{dir_name}/*")
    for f in files:
        os.remove(f)


def prepare_job(filename):
    lang = filename.split(".")[-1]
    if lang == "cpp":
        bin_name = filename.split(".")[0]
        fp = subprocess.run(
            ["g++"] + config.cpp_flags + ["-o", f"bin/{bin_name}", filename]
        )
        if fp.returncode != 0:
            raise RuntimeError("Compilation failed")
        return ["./bin/" + bin_name]
    elif lang == "py":
        return ["python3", filename]
    else:
        raise RuntimeError("Language is not supported: " + str(lang))


def add_test(name):
    subprocess.run(
        [
            "vim",
            "-o",
            os.path.join(input_dir, name + ".txt"),
            os.path.join(output_dir, name + ".txt"),
        ]
    )


def run_tests(job, tests):
    diff_max_length = config.runner["diff_max_length"]
    fout_name = config.runner["fout_name"]
    ferr_name = config.runner["ferr_name"]

    test_results = list()
    for filename in tests:
        tr = {"name": filename}
        with open(ferr_name, "w") as ferr:
            with open(fout_name, "w") as fout:
                with open(os.path.join(input_dir, filename), "r") as fin:
                    res = subprocess.run(job, stdin=fin, stdout=fout, stderr=ferr)
        tr["cout"] = get_head(fout_name, diff_max_length)
        tr["cerr"] = get_head(ferr_name, diff_max_length)
        if res.returncode != 0:
            tr["status"] = "RE"
        else:
            err, ann = diff_lines(
                tr["cout"],
                get_head(os.path.join(output_dir, filename), diff_max_length),
            )
            if err:
                tr["status"] = "WA"
                tr["cout"] = ann
            else:
                tr["status"] = "OK"
        test_results.append(tr)
    return test_results


def prep_parser():
    parser = argparse.ArgumentParser(description="Testing routine")
    for k, v in config.runner_parser.items():
        f, s = k.split(",")
        parser.add_argument(f, s, **v)
    parser.add_argument("filename", type=str, nargs="?")
    return parser.parse_args()


def reset_workspace(level: int):
    if level > 1:
        reset_default_sources()
    clean_dir(input_dir)
    clean_dir(output_dir)
    subprocess.run(["clear"])
    print("Updated at", datetime.datetime.now().time())


def infer_next_test_name():
    return str(sum(1 for filename in os.listdir(input_dir)))


def print_error_tests(test_results):
    for tr in test_results:
        if tr["status"] == "OK":
            continue
        print(f'-- {tr["name"]}: {tr["status"]}')
        print(
            "\n".join(
                get_head(
                    os.path.join(input_dir, tr["name"]),
                    config.runner["diff_max_length"],
                )
            )
        )
        print("-- got:")
        for line in tr["cout"]:
            print(line)
        print("-- expected:")
        print(
            "\n".join(
                get_head(
                    os.path.join(output_dir, tr["name"]),
                    config.runner["diff_max_length"],
                )
            )
        )
        print("-- cerr:")
        for line in tr["cerr"]:
            print(line)


def generate_test(gen: str, dumb: str):
    dumb_job = prepare_job(dumb)
    for line in fileinput.input(files=("-",)):
        test_name = infer_next_test_name() + ".txt"
        print(test_name)
        with open(os.path.join(input_dir, test_name), "w") as fin:
            subprocess.run([gen], input=line.encode(), stdout=fin)
        with open(os.path.join(output_dir, test_name), "w") as fout:
            with open(os.path.join(input_dir, test_name), "r") as fin:
                subprocess.run(dumb_job, stdin=fin, stdout=fout)


def main():
    args = prep_parser()

    if args.reset > 0:
        reset_workspace(args.reset)
        quit()

    if args.add_test > 0:
        test_name = args.filename
        if test_name is None:
            for _ in range(args.add_test):
                add_test(infer_next_test_name())
        else:
            add_test(test_name)
        quit()

    if args.gen:
        if args.dumb is None:
            print("Provide dumb solution to use")
            quit()
        generate_test(args.gen, args.dumb)
        quit()
    elif args.dumb:
        print("Provide generator")
        quit()

    source_filename = args.filename
    if args.filename is None:
        print("Provide source code to test...")
        quit()

    job = prepare_job(source_filename)

    tests = os.listdir(input_dir)
    if not args.specific_test is None:
        if len(args.specific_test) > 0:
            tests = set(tests) & set(x + ".txt" for x in args.specific_test)
        else:
            print("Running job only...")
            subprocess.run(job)
            quit()

    test_results = run_tests(job, tests)

    err_count = sum([1 for tr in test_results if tr["status"] != "OK"])
    if err_count > 0:
        print_error_tests(test_results)
        quit()

    print(f"See no evil 🙈 ({len(test_results)} tests passed)")
    with open(source_filename, "r") as source:
        subprocess.run(["pbcopy"], stdin=source)
    print("(solution copied to the clipboard)")


if __name__ == "__main__":
    main()

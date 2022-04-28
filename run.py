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

from test.colorize import err_style, colorize
from test.config import Config
from test.l10n import build_localizer


config = Config("stash/configs")
input_dir = config.runner["input_dir"]
perf_dir = config.runner["perf_dir"]
output_dir = config.runner["output_dir"]


Localizer = build_localizer(config.l10n)
l10n = Localizer()


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
    jp = os.path.join
    shutil.copyfile(jp("stash", "blank.cpp"), "main.cpp")
    open("easy.py", "w").close()


def clean_dirs(dir_names):
    for dir_name in dir_names:
        files = glob.glob(f"{dir_name}/*")
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
            else:
                shutil.rmtree(f)


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


def add_test(name):
    input_name = os.path.join(input_dir, f"{name}.txt")
    output_name = os.path.join(output_dir, f"{name}.txt")
    subprocess.run(["vim", "-o", input_name, output_name])


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


def reset_workspace(level):
    clean_dirs([input_dir, perf_dir, output_dir, config.runner["bin_dir"]])
    if level > 1:
        reset_default_sources()
    subprocess.run(["clear"])
    print(l10n.updated_at(datetime.datetime.now().time()))


def infer_next_test_name(dirname):
    return str(sum(1 for filename in os.listdir(dirname)))


def print_error_tests(test_results):
    head_length = config.runner["diff_max_length"]
    for tr in test_results:
        if tr["status"] == "OK":
            continue
        input_name = os.path.join(input_dir, tr["name"])
        output_name = os.path.join(output_dir, tr["name"])

        print(l10n.test_name_and_status(tr["name"], tr["status"]))
        print("\n".join(get_head(input_name, head_length)))
        print(l10n.test_got())
        for line in tr["cout"]:
            print(line)
        print(l10n.test_exp())
        print("\n".join(get_head(output_name, head_length)))
        print(l10n.test_cerr())
        for line in tr["cerr"]:
            print(line)


def run_generator(gen, output_path):
    generator_input = os.path.join(input_dir, "generator.txt")
    with open(generator_input, "w") as f:
        print(l10n.gen_input_prompt(gen), file=f)
    subprocess.run(["vim", "-o", generator_input, gen])
    gen_job = prepare_job(gen, perf=True)
    with open(generator_input, "r") as fin:
        with open(output_path, "w") as fout:
            subprocess.run(gen_job, stdin=fin, stdout=fout)
    os.remove(generator_input)


def generate_test(gen, dumb):
    test_name = f"{infer_next_test_name(input_dir)}.txt"
    run_generator(gen, os.path.join(input_dir, test_name))
    dumb_job = prepare_job(dumb, perf=True)
    with open(os.path.join(output_dir, test_name), "w") as fout:
        with open(os.path.join(input_dir, test_name), "r") as fin:
            subprocess.run(dumb_job, stdin=fin, stdout=fout)


def generate_perf(gen):
    test_name = f"{infer_next_test_name(perf_dir)}.txt"
    run_generator(gen, os.path.join(perf_dir, test_name))


def test_solution(source_filename, specific_test):
    job = prepare_job(source_filename, perf=False)
    tests = os.listdir(input_dir)
    if not specific_test is None:
        if len(specific_test) > 0:
            tests = set(tests) & set(x + ".txt" for x in specific_test)
        else:
            print(l10n.run_job_only())
            subprocess.run(job)
            return

    test_results = run_tests(job, tests)

    err_count = sum([1 for tr in test_results if tr["status"] != "OK"])
    if err_count > 0:
        print_error_tests(test_results)
        return

    print(l10n.tests_ok(len(test_results)))
    with open(source_filename, "r") as source:
        subprocess.run(["pbcopy"], stdin=source)
    print(l10n.solution_copied())


def perf_solution(source_filename):
    job = [config.runner["time_cmd"]] + prepare_job(source_filename, perf=True)
    tests = os.listdir(perf_dir)
    for filename in tests:
        print(colorize(filename, ["yellow"]))
        with open(os.path.join(perf_dir, filename), "r") as fin:
            devnull = subprocess.DEVNULL
            res = subprocess.run(job, stdin=fin, stdout=devnull)


def main():
    args = prep_parser()
    if args.reset > 0:
        reset_workspace(args.reset)
    elif args.add_test > 0:
        test_name = args.filename
        if test_name is None:
            for _ in range(args.add_test):
                add_test(infer_next_test_name(input_dir))
        else:
            add_test(test_name)
    elif args.gen:
        if args.dumb is None:
            generate_perf(args.gen)
        else:
            generate_test(args.gen, args.dumb)
    elif args.filename is None:
        print(err_style(l10n.no_filename_for_run()))
    elif args.perf > 0:
        perf_solution(args.filename)
    else:
        test_solution(args.filename, args.specific_test)


if __name__ == "__main__":
    main()

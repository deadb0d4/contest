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

from test.core import *


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
            exp = get_head(os.path.join(output_dir, filename), diff_max_length)
            err, ann = diff_lines(tr["cout"], exp)
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


def print_error_tests(test_results):
    head_length = config.runner["diff_max_length"]
    for tr in test_results:
        if tr["status"] == "OK":
            continue
        input_name = os.path.join(input_dir, tr["name"])
        output_name = os.path.join(output_dir, tr["name"])
        message = (
            [l10n.test_name_and_status(tr["name"], tr["status"])]
            + get_head(input_name, head_length)
            + [l10n.test_got()]
            + tr["cout"]
            + [l10n.test_exp()]
            + get_head(output_name, head_length)
            + [l10n.test_cerr()]
            + tr["cerr"]
        )
        for line in message:
            print(line)


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
    elif args.filename is None:
        print(err_style(l10n.no_filename_for_run()))
    else:
        test_solution(args.filename, args.specific_test)


if __name__ == "__main__":
    main()

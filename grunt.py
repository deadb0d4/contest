#!/usr/bin/python3

DESC = """
    Generator runs routine

    ➜ ./grunt.py -g <generator job>
        use generator to create perf case

    ➜ ./grunt.py -g <generator job> <dumb solution>
        use generator & dumb solution to create test case
        use runner to check against them

    ➜ ./grunt.py <solution name>
        run perf cases

"""
import argparse
import os
import subprocess

from test.core import *


def prep_parser():
    parser = argparse.ArgumentParser(
        description=DESC, formatter_class=argparse.RawTextHelpFormatter
    )
    for k, v in config.grunt_parser.items():
        f, s = k.split(",")
        parser.add_argument(f, s, **v)
    parser.add_argument("filename", type=str, nargs="?")
    return parser.parse_args()


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
    dumb_job = prepare_job(dumb, perf=False)
    with open(os.path.join(output_dir, test_name), "w") as fout:
        with open(os.path.join(input_dir, test_name), "r") as fin:
            subprocess.run(dumb_job, stdin=fin, stdout=fout)


def generate_perf(gen):
    test_name = f"{infer_next_test_name(perf_dir)}.txt"
    run_generator(gen, os.path.join(perf_dir, test_name))


def perf_solution(source_filename):
    print(warn_style("> Time error is ~50ms"))
    job = [config.runner["time_cmd"]] + prepare_job(source_filename, perf=True)
    tests = os.listdir(perf_dir)
    for filename in tests:
        print(warn_style(filename))
        with open(os.path.join(perf_dir, filename), "r") as fin:
            devnull = subprocess.DEVNULL
            res = subprocess.run(job, stdin=fin, stdout=devnull)


def main():
    args = prep_parser()
    if args.gen:
        if args.filename is None:
            generate_perf(args.gen)
        else:
            generate_test(args.gen, args.filename)
    elif args.filename is not None:
        perf_solution(args.filename)
    else:
        print(err_style(l10n.no_filename_for_run()))


if __name__ == "__main__":
    main()

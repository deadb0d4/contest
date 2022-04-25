#!/usr/bin/python3

import argparse
import collections
import datetime
import glob
import itertools
import os
import shutil
import subprocess


def get_head(filename, count):
    head = subprocess.run(
            ['head', f'-{count}', filename], stdout=subprocess.PIPE).stdout
    head = head.decode('utf-8').split('\n')
    while len(head) > 0 and head[-1] == '':
        head = head[:-1]
    return head

def colorize(x, cs):
    palette = {
            'red' : '\033[91m',
            'green' : '\033[92m',
            'yellow' : '\033[93m',
            'bold' : '\033[1m',
            'underline' : '\033[4m',
            'endc' : '\033[0m'
            }
    for c in cs:
        x = palette[c] + x + palette['endc']
    return x

def err_style(x):
    return colorize(x, ['red', 'bold'])

def tokenize(line):
    try:
        line = line.decode('utf-8')
    except:
        pass
    return list(line.split())

def diff_lines(lv, rv):
    err = False
    ann = []
    for l, r in itertools.zip_longest(lv, rv):
        if l is None:
            err = True
            res = err_style('(missing line)')
        elif r is None:
            err = True
            res = ' '.join(tokenize(l)) + err_style('\t (extra line)')
        else:
            lt, rt = map(tokenize, (l, r))
            if len(lt) != len(rt):
                err = True
                lt.append(err_style('\t (wrong length)'))
            else:
                for i in range(len(lt)):
                    if lt[i] != rt[i]:
                        err = True
                        lt[i] = err_style(lt[i])
            res = ' '.join(lt)
        ann.append(res)
    return err, ann

MAX_LENGTH = 100
TEST_DIR = 'test'
CACHE_DIR = 'stash'

buf_dir = os.path.join(TEST_DIR, 'buf')
input_dir = os.path.join(TEST_DIR, 'input')
output_dir = os.path.join(TEST_DIR, 'output')

fout_name = os.path.join(buf_dir, 'out.txt')
ferr_name = os.path.join(buf_dir, 'err.txt')


def reset_default_sources():
    shutil.copyfile(os.path.join(CACHE_DIR, 'blank.cpp'), 'main.cpp')
    open('easy.py', 'w').close()

def clean_dir(dir_name):
    files = glob.glob(f'{dir_name}/*')
    for f in files:
        os.remove(f)

def prepare_job(filename, lang):
    if lang == 'cpp':
        bin_name = filename.split('.')[0]
        fp = subprocess.run([
            'g++',
            '-std=c++17',
            '-g',
            '-Wshadow',
            '-Wuninitialized',
            '-DEBUG',
            '-I./stash/include/',  # missing includes
            '-fsanitize=address,undefined',
            '-o', 'bin/' + bin_name, filename])
        if fp.returncode != 0:
            raise RuntimeError('Compilation failed') 
        return ['./bin/' + bin_name]
    elif lang == 'py':
        return ['python3', filename]
    else:
        raise RuntimeError('Language is not supported: ' + str(lang))

def add_test(name):
    subprocess.run([
        'vim', '-o',
        os.path.join(input_dir, name + '.txt'),
        os.path.join(output_dir, name + '.txt')])

def run_tests(job, tests):
    test_results = list()
    for filename in tests:
        tr = {'name': filename}
        with open(ferr_name, 'w') as ferr:
            with open(fout_name, 'w') as fout:
                with open(os.path.join(input_dir, filename), 'r') as fin:
                   res = subprocess.run(job, stdin=fin, stdout=fout, stderr=ferr)
        tr['cout'] = get_head(fout_name, MAX_LENGTH)
        tr['cerr'] = get_head(ferr_name, MAX_LENGTH)
        if res.returncode != 0:
            tr['status'] = 'RE'
        else:
            err, ann = diff_lines(
                tr['cout'],
                get_head(os.path.join(output_dir, filename), MAX_LENGTH))
            if err:
                tr['status'] = 'WA'
                tr['cout'] = ann
            else:
                tr['status'] = 'OK'
        test_results.append(tr)
    return test_results

parser = argparse.ArgumentParser(description='Testing routine')
parser.add_argument(
        '--reset', '-r', action='count', help='reset workspace', default=0)
parser.add_argument(
        '--add_test', '-t', action='count', help='add test cases', default=0)
parser.add_argument(
        '--specific_test', '-s', nargs='*', help='use only selected tests')
parser.add_argument('filename', type=str, nargs='?')
args = parser.parse_args()


if args.reset > 0:
    if args.reset > 1:
        reset_default_sources()
    clean_dir(input_dir)
    clean_dir(output_dir)
    subprocess.run(['clear'])
    print('Updated at', datetime.datetime.now().time())
    quit()

if args.add_test > 0:
    test_name = args.filename
    if test_name is None:
        test_name = ''
        num = sum(1 for filename in os.listdir(input_dir))
        for i in range(args.add_test):
            add_test(test_name + str(num + i))
    else:
        add_test(test_name)
    quit()

source_filename = args.filename
if args.filename is None:
    print('Provide source code to test...')
    quit()

lang = source_filename.split('.')[-1]
job = prepare_job(source_filename, lang)

tests = os.listdir(input_dir)

if not args.specific_test is None:
    if len(args.specific_test) > 0:
        tests = set(tests) & set(x + '.txt' for x in args.specific_test)
    else:
        print('Running job only...')
        subprocess.run(job)
        quit()

test_results = run_tests(job, tests)

err_count = 0
for tr in test_results:
    if tr['status'] != 'OK':
        err_count += 1
        print(f'-- {tr["name"]}: {tr["status"]}')
        print('\n'.join(get_head(os.path.join(input_dir, tr['name']), MAX_LENGTH)))
        print('-- got:')
        for line in tr['cout']:
            print(line)
        print('-- expected:')
        print('\n'.join(get_head(os.path.join(output_dir, tr['name']), MAX_LENGTH)))
        print('-- cerr:')
        for line in tr['cerr']:
            print(line)

if err_count == 0:
    test_count = len(test_results)
    print(f'See no evil 🙈 ({test_count} tests passed)')

    with open(source_filename, 'r') as source:
        subprocess.run(['pbcopy'], stdin=source)

    print('(solution copied to the clipboard)')

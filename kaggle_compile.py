#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import sys
from typing import List

parser = argparse.ArgumentParser(
    description='Compile a list of python files into a Kaggle compatable script: \n' +
                './kaggle_compile.py [script_files.py] > output_file.py'
)
parser.add_argument('files', nargs='+',                                help='list of files to parse' )
parser.add_argument('--python-path', default='.',                      help='directory to search for local namespace imports')
parser.add_argument('--output-dir',  default='./data_output/scripts/', help='directory to write output if --save')
parser.add_argument('--save',        action='store_true',              help='should file be saved to disk')
args, unknown = parser.parse_known_args()  # Ignore extra CLI args passed in by Kaggle
if len(args.files) == 0:  parser.print_help(sys.stderr); sys.exit();


module_names = [ name for name in os.listdir(args.python_path)
                 if os.path.isdir(os.path.join(args.python_path, name))
                 and not name.startswith('.') ]
module_regex = '(?:' + "|".join(map(re.escape, module_names)) + ')'
import_regex = f'^from\s+({module_regex}.*?)\s+import'


def read_and_comment_file(filename: str) -> str:
    code = open(filename, 'r').read()
    code = re.sub(import_regex, r'# \0', code, re.M)
    return code


def extract_dependencies_from_file(filename: str) -> List[str]:
    code    = open(filename, 'r').read()
    imports = re.findall(import_regex, code, re.M)
    files   = list(map(lambda string: string.replace('.', '/')+'.py', imports))
    return files


def recurse_dependencies(filelist: List[str]) -> List[str]:
    output = filelist
    for filename in filelist:
        dependencies = extract_dependencies_from_file(filename)
        if len(dependencies):
            output = [
                recurse_dependencies(dependencies),
                dependencies,
                output
            ]
    output = flatten(output)
    return output


def flatten(filelist):
    output = []
    for item in filelist:
        if isinstance(item,list):
            if len(item):         output.extend(flatten(item))
        else:                     output.append(item)
    return output


def unique(filelist: List[str]) -> List[str]:
    seen   = {}
    output = []
    for filename in filelist:
        if not seen.get(filename, False):
            seen[filename] = True
            output.append(filename)
    return output


def make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2    # copy R bits to X
    os.chmod(path, mode)


def compile_script(filelist: List[str]) -> str:
    filelist = unique(filelist)
    gitinfo = [
        subprocess.check_output('date --rfc-3339 seconds',     shell=True),
        subprocess.check_output('git remote -v',               shell=True),
        subprocess.check_output('git branch -v ',              shell=True),
        subprocess.check_output('git rev-parse --verify HEAD', shell=True),
    ]
    gitinfo = "".join(map(lambda string: re.sub('^|\n+', '\n##### ', string.decode("utf-8"), re.MULTILINE), gitinfo))

    codes = ["#!/usr/bin/env python3"]
    codes.append(gitinfo)
    for filename in filelist:
        codes.append( f'#####\n##### START {filename}\n#####' )
        codes.append( read_and_comment_file(filename) )
        codes.append( f'#####\n##### END   {filename}\n#####' )
    codes.append(gitinfo)

    return "\n".join(codes)




if __name__ == '__main__':
    filenames = recurse_dependencies(args.files)
    code      = compile_script(filenames)
    print(code)
    if args.save:
        savefile = os.path.join( args.output_dir, os.path.basename(args.files[-1]) )  # Assume last provided filename
        open(savefile, 'w').write(code)
        make_executable(savefile)
        print('##### Wrote:', savefile, file=sys.stderr)
import os
import glob

test_files = glob.glob(os.path.join('tests', '*.py'))

for file in test_files:
    print(f"Running {file}...")
    os.system(f'python "{file}"')
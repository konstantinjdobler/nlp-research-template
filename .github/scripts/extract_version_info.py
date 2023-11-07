#!/usr/bin/env python3
import sys
import yaml

if len(sys.argv) != 2:
    print("Usage: python extract_pytorch_version.py <package_name>")
    sys.exit(1)

package_name = sys.argv[1]

# Load the lock file
try:
    with open('conda-lock.yml', 'r') as lock_file:
        lock_data = yaml.safe_load(lock_file)
except FileNotFoundError:
    print("Lock file 'conda-lock.yml' not found.")
    sys.exit(1)

# Extract the version of the specified package
package_version = None
for package in lock_data['package']:
    if package['name'] == package_name and package['platform'] == 'linux-64':
        package_version = package['version']
        break

if package_version:
    print(package_version)
else:
    print(f"{package_name}-not-found")

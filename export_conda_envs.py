#!/usr/bin/env python3
"""
Export conda environment information to create Docker-compatible requirements files.
"""

import subprocess
import json
import os

def get_conda_env_info(env_name):
    """Get detailed information about a conda environment"""
    try:
        # Get environment info
        result = subprocess.run(['conda', 'list', '-n', env_name, '--json'], 
                              capture_output=True, text=True, check=True)
        packages = json.loads(result.stdout)
        
        # Get Python version
        python_version = None
        for pkg in packages:
            if pkg['name'] == 'python':
                python_version = pkg['version']
                break
        
        return packages, python_version
    except subprocess.CalledProcessError as e:
        print(f"Error getting info for environment {env_name}: {e}")
        return None, None

def create_requirements_file(packages, output_file):
    """Create a requirements.txt file from conda package list"""
    pip_packages = []
    conda_packages = []
    
    for pkg in packages:
        name = pkg['name']
        version = pkg['version']
        channel = pkg.get('channel', 'unknown')
        
        # Skip base packages that don't need to be explicitly installed
        skip_packages = {'python', 'pip', 'setuptools', 'wheel', 'ca-certificates', 
                        'certifi', 'openssl', 'sqlite', 'tk', 'tzdata', 'xz', 'zlib'}
        
        if name in skip_packages:
            continue
            
        # Determine if it's a pip or conda package
        if channel == 'pypi':
            pip_packages.append(f"{name}=={version}")
        else:
            conda_packages.append(f"{name}=={version}")
    
    # Write requirements file
    with open(output_file, 'w') as f:
        f.write("# Conda packages\n")
        for pkg in sorted(conda_packages):
            f.write(f"# {pkg}\n")
        f.write("\n# Pip packages\n")
        for pkg in sorted(pip_packages):
            f.write(f"{pkg}\n")
    
    return len(conda_packages), len(pip_packages)

def main():
    environments = {
        'dlkcat_env': 'dlkcat',
        'eitlem_env': 'eitlem', 
        'turnup_env': 'turnup',
        'unikp': 'unikp'
    }
    
    # Create docker-requirements directory
    os.makedirs('docker-requirements', exist_ok=True)
    
    print("Exporting conda environments...")
    
    for env_name, short_name in environments.items():
        print(f"\nProcessing {env_name}...")
        
        packages, python_version = get_conda_env_info(env_name)
        if packages is None:
            print(f"Failed to get info for {env_name}")
            continue
            
        print(f"Python version: {python_version}")
        print(f"Total packages: {len(packages)}")
        
        # Create requirements file
        req_file = f"docker-requirements/{short_name}_requirements.txt"
        conda_count, pip_count = create_requirements_file(packages, req_file)
        
        # Create Python version file
        with open(f"docker-requirements/{short_name}_python_version.txt", 'w') as f:
            f.write(python_version)
        
        print(f"Created {req_file}")
        print(f"  - Conda packages: {conda_count}")
        print(f"  - Pip packages: {pip_count}")
        print(f"Created docker-requirements/{short_name}_python_version.txt")

if __name__ == '__main__':
    main()

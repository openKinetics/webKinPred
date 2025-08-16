#!/usr/bin/env python3
"""
Create clean requirements files for each environment with only essential packages.
"""

import subprocess
import json
import os

def get_essential_packages(env_name):
    """Get essential packages from conda environment"""
    try:
        result = subprocess.run(['conda', 'list', '-n', env_name, '--json'], 
                              capture_output=True, text=True, check=True)
        packages = json.loads(result.stdout)
        
        # Define essential packages for each environment
        essential_patterns = {
            'dlkcat_env': ['rdkit', 'torch', 'torchvision', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'requests'],
            'eitlem_env': ['torch', 'torchvision', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'biopython', 'transformers'],
            'turnup_env': ['rdkit', 'torch', 'torchvision', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib'],
            'unikp': ['torch', 'torchvision', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'transformers', 'sentencepiece', 'rdkit', 'openpyxl', 'tqdm']
        }
        
        # Get Python version
        python_version = None
        for pkg in packages:
            if pkg['name'] == 'python':
                python_version = pkg['version']
                break
        
        # Filter essential packages
        essential_packages = []
        patterns = essential_patterns.get(env_name, [])
        
        for pkg in packages:
            name = pkg['name']
            version = pkg['version']
            
            # Check if package matches any essential pattern
            for pattern in patterns:
                if pattern.lower() in name.lower():
                    essential_packages.append((name, version))
                    break
        
        return essential_packages, python_version
        
    except subprocess.CalledProcessError as e:
        print(f"Error getting info for environment {env_name}: {e}")
        return None, None

def create_clean_requirements(env_name, short_name, packages, python_version):
    """Create clean requirements.txt file"""
    req_file = f"docker-requirements/{short_name}_requirements.txt"
    
    with open(req_file, 'w') as f:
        f.write(f"# Requirements for {env_name} (Python {python_version})\n")
        f.write("# Essential packages only\n\n")
        
        for name, version in sorted(packages):
            f.write(f"{name}=={version}\n")
    
    # Create Python version file
    with open(f"docker-requirements/{short_name}_python_version.txt", 'w') as f:
        f.write(python_version)
    
    return req_file

def main():
    environments = {
        'dlkcat_env': 'dlkcat',
        'eitlem_env': 'eitlem', 
        'turnup_env': 'turnup',
        'unikp': 'unikp'
    }
    
    print("Creating clean requirements files...")
    
    for env_name, short_name in environments.items():
        print(f"\nProcessing {env_name}...")
        
        packages, python_version = get_essential_packages(env_name)
        if packages is None:
            print(f"Failed to get info for {env_name}")
            continue
            
        print(f"Python version: {python_version}")
        print(f"Essential packages: {len(packages)}")
        
        req_file = create_clean_requirements(env_name, short_name, packages, python_version)
        print(f"Created {req_file}")
        
        # Print packages for verification
        for name, version in packages:
            print(f"  - {name}=={version}")

if __name__ == '__main__':
    main()

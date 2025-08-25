#!/usr/bin/env python3
"""
Project organization utility script.
This script helps organize the project structure and provides information about the current organization.
"""

import os
import shutil
from pathlib import Path
import config

def show_project_structure():
    """Display the current project structure."""
    print("=" * 60)
    print("CURRENT PROJECT STRUCTURE")
    print("=" * 60)
    
    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        if current_depth > max_depth:
            return
        
        try:
            items = sorted(os.listdir(directory))
            for i, item in enumerate(items):
                if item.startswith('.'):
                    continue
                    
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                next_prefix = "    " if is_last else "│   "
                
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    print(f"{prefix}{current_prefix}{item}/")
                    print_tree(item_path, prefix + next_prefix, max_depth, current_depth + 1)
                else:
                    print(f"{prefix}{current_prefix}{item}")
        except PermissionError:
            print(f"{prefix}└── [Permission Denied]")
    
    print_tree(".", max_depth=3)

def count_files_by_type():
    """Count files by type in different directories."""
    print("\n" + "=" * 60)
    print("FILE COUNT BY TYPE")
    print("=" * 60)
    
    directories = {
        "Python Scripts": ["analysis"],
        "Images": ["outputs/figures"],
        "Data Files": ["data"],
        "Logs": ["outputs/logs"],
        "Results": ["outputs/figures/results"]
    }
    
    for category, dirs in directories.items():
        total_files = 0
        file_types = {}
        
        for dir_path in dirs:
            if os.path.exists(dir_path):
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        if not file.startswith('.'):
                            total_files += 1
                            ext = Path(file).suffix.lower()
                            file_types[ext] = file_types.get(ext, 0) + 1
        
        print(f"\n{category}:")
        print(f"  Total files: {total_files}")
        if file_types:
            for ext, count in sorted(file_types.items()):
                print(f"  {ext}: {count}")

def check_organization():
    """Check if the project is properly organized."""
    print("\n" + "=" * 60)
    print("ORGANIZATION CHECK")
    print("=" * 60)
    
    issues = []
    
    # Check if Python files are in the right places
    root_py_files = [f for f in os.listdir('.') if f.endswith('.py') and f not in ['config.py', 'organize_project.py']]
    if root_py_files:
        issues.append(f"Python files still in root: {root_py_files}")
    
    # Check if images are organized
    if os.path.exists('outputs/figures'):
        figure_dirs = [d for d in os.listdir('outputs/figures') if os.path.isdir(os.path.join('outputs/figures', d))]
        if not figure_dirs:
            issues.append("No figure subdirectories found")
    else:
        issues.append("Figures directory not found")
    
    # Check if data is organized
    if not os.path.exists('data/metrics'):
        issues.append("Metrics directory not found")
    
    if issues:
        print("❌ Organization issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Project appears to be well organized!")

def create_missing_directories():
    """Create any missing directories."""
    print("\n" + "=" * 60)
    print("CREATING MISSING DIRECTORIES")
    print("=" * 60)
    
    config.ensure_directories()
    print("✅ All directories created/verified!")

def main():
    """Main function to run the organization utility."""
    print("🧠 Neuroscience Simulation Project Organization Utility")
    print("=" * 60)
    
    # Create missing directories
    create_missing_directories()
    
    # Show current structure
    show_project_structure()
    
    # Count files
    count_files_by_type()
    
    # Check organization
    check_organization()
    
    print("\n" + "=" * 60)
    print("ORGANIZATION COMPLETE!")
    print("=" * 60)
    print("\nYour project is now organized with the following structure:")
    print("📁 analysis/     - Python scripts organized by category")
    print("📁 data/         - Data files and metrics")
    print("📁 outputs/      - Generated figures, reports, and logs")
    print("📁 brian2/       - Brian2 library source")
    print("📁 universal/    - Universal number system library")
    
    print("\nNext steps:")
    print("1. Update import paths in your scripts if needed")
    print("2. Run your analysis scripts from their new locations")
    print("3. Check that outputs are going to the correct directories")

if __name__ == "__main__":
    main() 
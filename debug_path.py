#!/usr/bin/env python3
"""
Debug Python path issues
"""
import os
import sys

# Test different path approaches
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Project root: {project_root}")

# Test 1: Direct path
direct_path = os.path.join(project_root, 'src')
sys.path.insert(0, direct_path)
print(f"Direct path: {direct_path}")

try:
    import src.pipeline
    print("✅ Direct path import works!")
except ImportError as e:
    print(f"❌ Direct path import failed: {e}")

# Test 2: Relative path
relative_path = os.path.join(project_root, 'src')
sys.path.insert(0, relative_path)
print(f"Relative path: {relative_path}")

try:
    from src.pipeline import create_pipeline
    print("✅ Relative path import works!")
except ImportError as e:
    print(f"❌ Relative path import failed: {e}")

# Test 3: Package import
sys.path.insert(0, project_root)
print(f"Package path: {project_root}")

try:
    import src.pipeline
    print("✅ Package import works!")
except ImportError as e:
    print(f"❌ Package import failed: {e}")

print(f"Python path: {sys.path}")
print("🎉 Debug test completed!")

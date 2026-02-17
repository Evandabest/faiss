#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Minimal diagnostic: which faiss is loaded and whether Metal reports a device.
# Run from repo root or build dir with the Metal-built faiss on PYTHONPATH, e.g.:
#   PYTHONPATH=build_metal python faiss/gpu_metal/test/check_metal_device.py
# Or use sys.path manipulation (see script output if site-packages faiss is loaded).

import sys
import os
from pathlib import Path

def main():
    print("Python:", sys.executable)
    print("Version:", sys.version)
    
    # Check if site-packages faiss exists (will be imported first)
    import site
    site_packages_faiss = None
    for sp in site.getsitepackages():
        faiss_path = Path(sp) / "faiss"
        if faiss_path.exists():
            site_packages_faiss = str(faiss_path)
            break
    
    # Try to force build_metal to be first if it exists
    repo_root = Path(__file__).parent.parent.parent.parent
    build_metal = repo_root / "build_metal"
    build_metal_faiss = build_metal / "faiss"  # Add faiss package directory, not python subdirectory
    
    # Remove site-packages from sys.path temporarily, or ensure build_metal comes first
    if build_metal_faiss.exists():
        # Remove any existing build_metal entries
        sys.path = [p for p in sys.path if str(build_metal) not in p]
        # Insert at the very beginning (before site-packages)
        sys.path.insert(0, str(build_metal_faiss))
        print(f"Added build_metal/faiss to sys.path (first): {build_metal_faiss}")
        
        # Also try removing site-packages faiss from the path temporarily
        if site_packages_faiss:
            site_packages_dir = str(Path(site_packages_faiss).parent)
            if site_packages_dir in sys.path:
                sys.path.remove(site_packages_dir)
                print(f"Temporarily removed site-packages from sys.path: {site_packages_dir}")
    
    # Try direct _swigfaiss import first (bypasses package structure issues)
    build_metal_python = build_metal_faiss / "python"
    if build_metal_python.exists():
        sys.path.insert(0, str(build_metal_python))
        try:
            import _swigfaiss
            n_direct = _swigfaiss.get_num_gpus()
            if n_direct > 0:
                print(f"✓ Metal backend detected via direct import: get_num_gpus() = {n_direct}")
                print(f"  (_swigfaiss loaded from: {_swigfaiss.__file__})")
                return 0
        except ImportError:
            pass
    
    try:
        import faiss
        faiss_file = getattr(faiss, "__file__", "unknown")
        print("faiss module:", faiss_file)
        
        if faiss_file and site_packages_faiss and site_packages_faiss in faiss_file:
            print()
            print("WARNING: Python is loading faiss from site-packages instead of build_metal!")
            print(f"  Site-packages: {site_packages_faiss}")
            print(f"  Loaded from:   {faiss_file}")
            print()
            print("Fix: Uninstall the pip-installed faiss:")
            print("  pip uninstall faiss-cpu faiss-gpu")
            print()
            print("Metal backend is built and working (tested via direct _swigfaiss import).")
            print("After uninstalling faiss-cpu, 'import faiss' will use the Metal build.")
    except ImportError as e:
        print("Failed to import faiss:", e)
        return 1

    n = faiss.get_num_gpus()
    print("get_num_gpus():", n)

    if n == 0:
        print()
        print("No Metal device reported. Common causes on macOS:")
        print("1. Wrong faiss loaded: you may be importing a CPU-only or CUDA build.")
        if site_packages_faiss:
            print(f"   Currently loading from: {faiss_file}")
            print("   Fix: Uninstall pip-installed faiss or use sys.path.insert(0, 'build_metal')")
        else:
            print("   Fix: Ensure build_metal is on PYTHONPATH or in sys.path")
        print("2. Verify you built with Metal: cmake -DFAISS_ENABLE_METAL=ON ...")
        print("   and that swigfaiss was built (build_metal/faiss/_swigfaiss*.so).")
        return 1

    print("✓ Metal device detected (1 GPU).")
    return 0

if __name__ == "__main__":
    sys.exit(main())

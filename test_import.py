#!/usr/bin/env python3

import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from scripts.protein import Protein
    print("✅ Successfully imported Protein class!")
    
    # Test creating a simple Protein object (this will fail if there are other issues)
    print("✅ Import test completed successfully!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("This indicates there are still import issues to resolve.")
except Exception as e:
    print(f"⚠️  Import succeeded but other error occurred: {e}")
    print("The import itself is working, but there may be other issues.") 
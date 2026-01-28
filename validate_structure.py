#!/usr/bin/env python3
"""
Simple structure validation script.
Checks that all modules can be imported and basic structure is correct.
"""
import sys
import importlib.util

def check_module(path, module_name):
    """Check if a module can be loaded."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return False, f"Could not load spec for {module_name}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, f"{module_name}: OK"
    except SyntaxError as e:
        return False, f"{module_name}: Syntax error - {e}"
    except Exception as e:
        return False, f"{module_name}: Error - {e}"

def main():
    """Validate project structure."""
    checks = [
        ("data/synthetic.py", "synthetic"),
        ("data/loader.py", "loader"),
        ("data/graph.py", "graph"),
        ("models/fahgt_t.py", "fahgt_t"),
        ("models/sequence_encoder.py", "sequence_encoder"),
        ("models/fusion_head.py", "fusion_head"),
        ("models/model.py", "model"),
        ("losses/risk_aware.py", "risk_aware"),
        ("training/trainer.py", "trainer"),
        ("training/metrics.py", "metrics"),
        ("federated/client.py", "client"),
        ("federated/server.py", "server"),
        ("federated/partition.py", "partition"),
    ]
    
    all_ok = True
    for path, name in checks:
        ok, msg = check_module(path, name)
        print(msg)
        if not ok:
            all_ok = False
    
    if all_ok:
        print("\n✓ All modules have valid syntax!")
        print("Note: Runtime import errors are expected if dependencies are not installed.")
        print("Install dependencies with: pip install -r requirements.txt")
    else:
        print("\n✗ Some modules have issues. Please fix syntax errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()


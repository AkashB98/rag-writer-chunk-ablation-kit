#!/usr/bin/env python3
"""Run the full hermetic test suite (stdlib unittest)."""
import sys, unittest

if __name__ == "__main__":
    suite = unittest.TestLoader().discover("tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"tests run: {result.testsRun}, failures: {len(result.failures)}, errors: {len(result.errors)}")
    sys.exit(0 if result.wasSuccessful() else 1)

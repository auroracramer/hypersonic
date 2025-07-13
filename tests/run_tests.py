#!/usr/bin/env python3
"""
Test runner script for the video understanding project.
This script provides easy ways to run different test suites.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_command(cmd):
    """Run a command and return its result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run tests for the video understanding project')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--coverage', action='store_true', help='Run tests with coverage report')
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fast', action='store_true', help='Run fast tests only (skip slow tests)')
    parser.add_argument('--module', type=str, help='Run tests for specific module (e.g., models, losses)')
    parser.add_argument('--gpu', action='store_true', help='Run GPU tests')
    parser.add_argument('--no-gpu', action='store_true', help='Skip GPU tests')
    parser.add_argument('--html', action='store_true', help='Generate HTML coverage report')
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd_parts = ['python', '-m', 'pytest']
    
    # Add verbose flag
    if args.verbose:
        cmd_parts.append('-v')
    
    # Add parallel execution
    if args.parallel:
        cmd_parts.extend(['-n', 'auto'])
    
    # Add coverage
    if args.coverage:
        cmd_parts.extend(['--cov=.', '--cov-report=term-missing'])
        if args.html:
            cmd_parts.append('--cov-report=html')
    
    # Test selection
    if args.unit:
        cmd_parts.extend(['-m', 'unit'])
    elif args.integration:
        cmd_parts.extend(['-m', 'integration'])
    elif args.fast:
        cmd_parts.extend(['-m', 'not slow'])
    elif args.gpu:
        cmd_parts.extend(['-m', 'gpu'])
    elif args.no_gpu:
        cmd_parts.extend(['-m', 'not gpu'])
    
    # Module-specific tests
    if args.module:
        cmd_parts.append(f'tests/test_{args.module}.py')
    
    # Default to all tests if no specific selection
    if not any([args.unit, args.integration, args.fast, args.gpu, args.no_gpu, args.module]):
        cmd_parts.append('tests/')
    
    # Run the command
    cmd = ' '.join(cmd_parts)
    return run_command(cmd)


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking test dependencies...")
    
    try:
        import pytest
        import torch
        import numpy as np
        import pandas as pd
        from PIL import Image
        print("✓ All required dependencies found")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install test dependencies:")
        print("pip install -r tests/requirements.txt")
        return False


def run_quick_check():
    """Run a quick smoke test to verify the setup."""
    print("Running quick smoke test...")
    cmd = "python -m pytest tests/test_utils.py::TestAverageMeter::test_average_meter_init -v"
    result = run_command(cmd)
    if result == 0:
        print("✓ Quick check passed")
    else:
        print("✗ Quick check failed")
    return result


def run_coverage_report():
    """Run tests with coverage and generate report."""
    print("Running tests with coverage...")
    cmd = ("python -m pytest tests/ --cov=. --cov-report=html --cov-report=term-missing "
           "--cov-report=xml --cov-fail-under=70")
    result = run_command(cmd)
    if result == 0:
        print("✓ Coverage report generated in htmlcov/")
    return result


def run_performance_tests():
    """Run performance-related tests."""
    print("Running performance tests...")
    cmd = "python -m pytest tests/ -k 'performance or speed or memory' -v"
    return run_command(cmd)


if __name__ == '__main__':
    # Check if running with specific commands
    if len(sys.argv) > 1:
        if sys.argv[1] == 'check':
            sys.exit(0 if check_dependencies() else 1)
        elif sys.argv[1] == 'quick':
            sys.exit(run_quick_check())
        elif sys.argv[1] == 'coverage':
            sys.exit(run_coverage_report())
        elif sys.argv[1] == 'performance':
            sys.exit(run_performance_tests())
        elif sys.argv[1] == 'help':
            print(__doc__)
            print("\nUsage examples:")
            print("  python tests/run_tests.py check          # Check dependencies")
            print("  python tests/run_tests.py quick          # Quick smoke test")
            print("  python tests/run_tests.py coverage       # Run with coverage")
            print("  python tests/run_tests.py performance    # Run performance tests")
            print("  python tests/run_tests.py --module models # Test specific module")
            print("  python tests/run_tests.py --fast         # Skip slow tests")
            print("  python tests/run_tests.py --parallel     # Run in parallel")
            sys.exit(0)
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Run main test suite
    sys.exit(main())
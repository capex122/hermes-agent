#!/usr/bin/env python3
"""Run anti-bot tests 3 times and analyze cumulative results."""

import json
import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime

def run_test(run_num: int):
    """Run the anti-bot test suite once."""
    print(f"\n{'='*70}")
    print(f"RUN {run_num}/3 - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")
    
    output_file = f".tmp/antibot-results-run{run_num}.json"
    cmd = [
        sys.executable, "scripts/test_antibot_sites.py",
        "--mode", "search",
        "--output-json", output_file
    ]
    
    env = os.environ.copy()
    env["HERMES_BROWSER_BOT_RETRIES"] = "3"
    env["PYTHONPATH"] = os.getcwd()
    
    result = subprocess.run(cmd, env=env, cwd=os.getcwd())
    
    if result.returncode == 0 and Path(output_file).exists():
        with open(output_file) as f:
            data = json.load(f)
        successes = [s['success'] for s in data]
        pass_rate = sum(successes) / len(successes) * 100
        print(f"Pass Rate: {sum(successes)}/{len(successes)} ({pass_rate:.1f}%)")
        
        for s in data:
            site = s['site'].split('://')[-1].split('/')[0]
            path = '/'.join(s['site'].split('://')[-1].split('/')[1:])
            display = f"{site}/{path}" if path and path != '' else site
            status = "✓" if s['success'] else "✗"
            print(f"  {status} {display}")
        
        return data
    else:
        print("Test run FAILED")
        return None

def analyze_results(run1, run2, run3):
    """Analyze if all sites passed all 3 times."""
    print(f"\n{'='*70}")
    print("CUMULATIVE ANALYSIS - All 3 Runs")
    print(f"{'='*70}\n")
    
    # Get unique sites
    sites = [s['site'] for s in run1]
    
    all_pass_3x = True
    for site in sites:
        r1 = next((s['success'] for s in run1 if s['site'] == site), False)
        r2 = next((s['success'] for s in run2 if s['site'] == site), False)
        r3 = next((s['success'] for s in run3 if s['site'] == site), False)
        
        passed = sum([r1, r2, r3])
        display = site.split('://')[-1].split('/')[0]
        status = "✓✓✓" if (r1 and r2 and r3) else f"{passed}/3"
        
        print(f"  {status:6} {display}")
        if not (r1 and r2 and r3):
            all_pass_3x = False
    
    total_runs = len(sites) * 3
    total_passed = sum([
        sum([s['success'] for s in run1]),
        sum([s['success'] for s in run2]),
        sum([s['success'] for s in run3]),
    ])
    
    print(f"\nTotal: {total_passed}/{total_runs} ({total_passed/total_runs*100:.1f}%)")
    print(f"Target: 8/8 all 3 runs = 24/24 (100%)\n")
    
    if all_pass_3x:
        print("✓✓✓ SUCCESS! All 8 sites passed 3 times in a row! ✓✓✓")
    else:
        print(f"Progress: {total_passed}/24 tests passed")
    
    return all_pass_3x

if __name__ == "__main__":
    print("Starting 3x anti-bot test suite...")
    
    run1 = run_test(1)
    if not run1:
        sys.exit(1)
    
    run2 = run_test(2)
    if not run2:
        sys.exit(1)
    
    run3 = run_test(3)
    if not run3:
        sys.exit(1)
    
    success = analyze_results(run1, run2, run3)
    sys.exit(0 if success else 1)

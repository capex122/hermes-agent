#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def analyze_results(json_file):
    with open(json_file) as f:
        data = json.load(f)
    
    successes = [s['success'] for s in data]
    pass_rate = sum(successes) / len(successes) * 100
    
    print(f"\n{'='*70}")
    print(f"Pass Rate: {sum(successes)}/{len(successes)} ({pass_rate:.1f}%)")
    print(f"{'='*70}")
    
    for s in data:
        site = s['site'].split('://')[-1].split('/')[0]
        path = '/'.join(s['site'].split('://')[-1].split('/')[1:])
        display = f"{site}/{path}" if path else site
        status = "✓ PASS" if s['success'] else "✗ FAIL"
        retries = f" ({s['fresh_session_retry_count']} retries)" if s['fresh_session_retry_count'] > 0 else ""
        challenge = f" [{s['challenge_pattern']}]" if s.get('challenge_pattern') else ""
        
        print(f"  {status:8} {display:50} {retries}{challenge}")
    
    return sum(successes), len(successes)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = ".tmp/antibot-results-enhanced-v1.json"
    
    analyze_results(json_file)

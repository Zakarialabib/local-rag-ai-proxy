#!/usr/bin/env python
"""Quick endpoint validation tests."""

from webapp.app import create_app
import json

def test_endpoints():
    app = create_app()
    
    with app.test_client() as client:
        tests = [
            ('GET', '/', 'Dashboard'),
            ('GET', '/api/state', 'API State'),
            ('GET', '/api/acid/sessions', 'ACID Sessions'),
            ('POST', '/api/runtime/refresh', 'Runtime Refresh'),
        ]
        
        for method, path, name in tests:
            try:
                if method == 'GET':
                    resp = client.get(path)
                else:
                    resp = client.post(path, json={})
                
                status = '✓' if resp.status_code in [200, 201] else '✗'
                print(f"{status} {name:25} {method:4} {path:30} → {resp.status_code}")
                
                if resp.status_code == 200 and method == 'GET':
                    try:
                        data = resp.get_json()
                        if data and isinstance(data, dict):
                            keys = list(data.keys())[:3]
                            print(f"  └─ Keys: {keys}")
                    except:
                        pass
            except Exception as e:
                print(f"✗ {name:25} {method:4} {path:30} → ERROR: {str(e)[:40]}")

if __name__ == '__main__':
    print("\n=== ENDPOINT VALIDATION TEST ===\n")
    test_endpoints()
    print("\n=== TESTS COMPLETE ===\n")

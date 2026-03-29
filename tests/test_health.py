#!/usr/bin/env python
"""Test health and perfection endpoints."""

from webapp.app import create_app

def test_health_endpoints():
    app = create_app()
    
    with app.test_client() as client:
        tests = [
            ('GET', '/api/tools/health', 'Tool Health'),
            ('GET', '/api/tools/perfection', 'Tool Perfection'),
            ('GET', '/api/tools/health/timeline', 'Health Timeline'),
            ('POST', '/api/tools/perfection/reset', 'Reset Perfection'),
        ]
        
        for method, path, name in tests:
            try:
                if method == 'GET':
                    resp = client.get(path)
                else:
                    resp = client.post(path, json={})
                
                status = '✓' if resp.status_code in [200, 201] else '✗'
                print(f"{status} {name:25} {method:4} {path:35} → {resp.status_code}")
                
                if resp.status_code in [200, 201]:
                    try:
                        data = resp.get_json()
                        if "ok" in data:
                            print(f"  └─ ok={data['ok']}")
                        elif "error" in data:
                            print(f"  └─ error={data['error'][:50]}")
                    except:
                        pass
            except Exception as e:
                print(f"✗ {name:25} {method:4} {path:35} → ERROR: {str(e)[:40]}")

if __name__ == '__main__':
    print("\n=== HEALTH & PERFECTION ENDPOINT TESTS ===\n")
    test_health_endpoints()
    print("\n=== TESTS COMPLETE ===\n")

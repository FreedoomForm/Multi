#!/usr/bin/env python3
"""Submit a .cu file to the Tensara GPU sandbox (Tesla T4) and stream results.

Usage:  python3 run_sandbox.py <file.cu>
Auth:   set TENSARA_SESSION env var to the __Secure-next-auth.session-token
        (falls back to the token used for this project).
"""
import json
import os
import sys
import urllib.request

SESSION = os.environ.get("TENSARA_SESSION", "4a54b16a-715f-4258-a35c-5d4dd7f51648")
CSRF = ("c3e60e99ff5833fdf1cc648569db3bf7fc3d7f2ef94d7e2bb0383794959fb8f1"
        "%7Cb0ba64ab2f82c5aca56e0f15e82ab8e15419e71a7e9876a2479cb59178763ae2")
COOKIE = (f"__Host-next-auth.csrf-token={CSRF}; "
          f"__Secure-next-auth.callback-url=https%3A%2F%2Ftensara.org%2Fproblems; "
          f"__Secure-next-auth.session-token={SESSION}")


def run(path: str, language: str = "cuda"):
    code = open(path).read()
    req = urllib.request.Request(
        "https://tensara.org/api/submissions/sandbox",
        data=json.dumps({"code": code, "language": language}).encode(),
        headers={
            "Content-Type": "application/json",
            "Cookie": COOKIE,
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://tensara.org/sandbox/freedoomform/til",
        })
    rc = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            try:
                d = json.loads(line[5:])
            except Exception:
                continue
            st = d.get("status", "")
            if st == "SANDBOX_OUTPUT":
                print(f"[{d.get('stream')}] {d.get('line')}", flush=True)
            elif st == "SANDBOX_ERROR":
                print(f"== ERROR: {d.get('message', '')}", flush=True)
                det = d.get("details")
                if det:
                    print(str(det)[:6000], flush=True)
            elif st == "COMPILE_ERROR":
                print("== COMPILE_ERROR", flush=True)
                det = d.get("details") or d.get("message")
                if det:
                    print(str(det)[:6000], flush=True)
            elif st == "SANDBOX_SUCCESS":
                rc = d.get("return_code")
                print(f"== SUCCESS rc={rc}", flush=True)
            else:
                print(f"== {st}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(0 if run(sys.argv[1]) == 0 else 1)

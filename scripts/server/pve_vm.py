"""Scoped Proxmox control for Zeke's desktop VMs (101 windows10, 102 zorin) via token iris@pve!vmctl.
Secrets in ~/.iris_private.env (PVE_TOKEN_ID / PVE_TOKEN_SECRET / PVE_HOST). Zeke's rule: only ONE of 101/102 runs.
usage: pve_vm.py status | start <vmid> | shutdown <vmid> | exec <vmid> <powershell or bash command...>
"""
import json, os, ssl, sys, time, urllib.parse, urllib.request

env = {}
for line in open(os.path.expanduser("~/.iris_private.env")):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1); env[k] = v
BASE = f"https://{env['PVE_HOST']}:8006/api2/json/nodes/r740/qemu"
HDR = {"Authorization": f"PVEAPIToken={env['PVE_TOKEN_ID']}={env['PVE_TOKEN_SECRET']}"}
CTX = ssl._create_unverified_context()


def call(path, method="GET", data=None):
    body = urllib.parse.urlencode(data, doseq=True).encode() if data else None
    req = urllib.request.Request(BASE + path, data=body, method=method, headers=HDR)
    with urllib.request.urlopen(req, context=CTX, timeout=60) as r:
        return json.load(r)["data"]


def status(v):
    return call(f"/{v}/status/current")["status"]


def run(v, cmd, timeout=900):
    shell = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command"] if v == 101 else ["bash", "-c"]
    pid = call(f"/{v}/agent/exec", "POST", {"command": shell + [cmd]})["pid"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = call(f"/{v}/agent/exec-status?pid={pid}")
        if d.get("exited"):
            return d.get("exitcode"), d.get("out-data", ""), d.get("err-data", "")
        time.sleep(3)
    return None, "", "timeout"


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "status":
        for v in (101, 102):
            print(v, status(v))
    elif a[0] == "start":
        v = int(a[1]); other = 102 if v == 101 else 101
        if status(other) == "running":
            sys.exit(f"refusing: VM {other} is running (Zeke's one-at-a-time rule) — ask him first")
        print(call(f"/{v}/status/start", "POST"))
    elif a[0] == "shutdown":
        print(call(f"/{int(a[1])}/status/shutdown", "POST"))
    elif a[0] == "exec":
        code, out, err = run(int(a[1]), " ".join(a[2:]))
        print(out); print(err, file=sys.stderr); sys.exit(code or 0)

"""Build report for scene builders (pure python + numpy; no bpy). Replaces boss v2/v3's log()/tick()/REPORT dicts.

    from lib.report import Report
    with Report("/ABS/out/build_report.json", "bossfight v4") as R:    # ALWAYS saved — also when the build raises
        R.log("terrain", n_tris)                 # print + keep
        R.tick("terrain")                        # seconds since the previous tick → timings
        R.add("hoard.coins", 1834)               # dotted keys nest
        R.assert_(dmin > 0.3, "cam_clearance", f"camera {dmin:.2f} m from geometry")   # recorded; raises on failure
        R.check(...)                             # same, but only records (soft)
The saved JSON also carries lib/importer's import log + every file loaded ("imports", "files") when the importer was
used, so it can be fed straight to lib/assets/credits.py --used build_report.json.
"""
import datetime
import json
import os
import sys
import time
import traceback


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    if hasattr(o, "name"):                      # bpy ID
        return getattr(o, "name")
    return str(o)


class Report:
    def __init__(self, path=None, name="", quiet=False):
        self.path = path
        self.quiet = quiet
        self.data = {"name": name, "started": datetime.datetime.now().isoformat(timespec="seconds"),
                     "status": "running", "timings": {}, "values": {}, "asserts": [], "log": []}
        self._t0 = self._last = time.time()

    # ---- recording
    def log(self, *a):
        msg = " ".join(str(x) for x in a)
        self.data["log"].append(msg)
        if not self.quiet:
            print(f"[{self.data['name'] or 'build'}] {msg}", flush=True)

    def tick(self, label):
        now = time.time()
        self.data["timings"][label] = round(now - self._last, 2)
        self._last = now
        self.log(f"{label}: {self.data['timings'][label]} s")
        return self.data["timings"][label]

    def add(self, key, val):
        d = self.data["values"]
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
        return val

    def get(self, key, default=None):
        d = self.data["values"]
        for p in key.split("."):
            if not isinstance(d, dict) or p not in d:
                return default
            d = d[p]
        return d

    def check(self, cond, key, msg=""):
        """soft: record pass/fail, return bool"""
        ok = bool(cond)
        self.data["asserts"].append({"key": key, "ok": ok, "msg": msg})
        self.log(f"{'PASS' if ok else 'FAIL'} {key} {msg}")
        return ok

    def assert_(self, cond, key, msg=""):
        """hard: record, and on failure save the report and raise AssertionError"""
        if not self.check(cond, key, msg):
            self.data["status"] = f"failed: {key}"
            self.save()
            raise AssertionError(f"{key}: {msg}")
        return True

    @property
    def failed(self):
        return [a for a in self.data["asserts"] if not a["ok"]]

    # ---- output
    def save(self, path=None):
        path = path or self.path
        if not path:
            return None
        self.data["elapsed_s"] = round(time.time() - self._t0, 1)
        imp = sys.modules.get("lib.importer") or sys.modules.get("importer")
        if imp is not None and hasattr(imp, "LOG"):
            self.data["imports"] = imp.LOG
            self.data["files"] = sorted(getattr(imp, "USED_FILES", ()))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1, default=_jsonable)
        os.replace(tmp, path)
        return path

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is not None and not str(self.data["status"]).startswith("failed"):
            self.data["status"] = f"error: {et.__name__}: {ev}"
            self.data["traceback"] = "".join(traceback.format_exception(et, ev, tb))[-4000:]
        elif et is None:
            self.data["status"] = "ok" if not self.failed else f"ok with {len(self.failed)} soft failures"
        self.save()
        return False

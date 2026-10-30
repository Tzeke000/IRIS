#!/bin/sh
# Demote DISPOSABLE helpers inside the OOM-protected core units back to oom_score_adj 0.
# iris-oom-protect (root, Zeke 10-11) sets -1000 on every proc in iris-stack/mouth/ears at unit start,
# and children INHERIT it -> the Fish mouth's 32 torch-inductor compile workers were OOM-immune too
# (10-30 V3 Phase 0 baseline: 56 procs at -1000, 34 of them compile workers). The core processes stay
# at -1000; only known-disposable helpers go back to 0. Raising a same-uid score needs no privilege.
B=/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice
n=0
for u in iris-stack iris-mouth iris-ears; do
  CG=$B/$u.service/cgroup.procs; [ -r "$CG" ] || continue
  for p in $(cat "$CG"); do
    a=$(cat /proc/$p/oom_score_adj 2>/dev/null) || continue
    [ "$a" -lt 0 ] 2>/dev/null || continue
    case "$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)" in
      *torch/_inductor/compile_worker*) echo 0 > /proc/$p/oom_score_adj 2>/dev/null && n=$((n+1));;
    esac
  done
done
[ "$n" -gt 0 ] && echo "demoted $n compile workers to oom 0"
exit 0

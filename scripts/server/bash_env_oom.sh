# BASH_ENV for my Claude session (set by iris_body_host.py) - sourced by every NON-interactive bash the
# CLI spawns (the Bash tool, scripts, workers). 10-29 code review: oom_score_adj is INHERITED across fork,
# so with the stack at -1000 (iris-oom-protect, Zeke 10-11) every shell and worker I started was OOM-immune
# too - under memory pressure the kernel had nothing it was allowed to kill (the freeze). Core processes
# stay at -1000; my disposable children go back to 0 here. Raising one's own score needs no privilege.
if [ -n "${CLAUDECODE:-}" ] && [ -w /proc/self/oom_score_adj ]; then
  read -r _iris_adj < /proc/self/oom_score_adj 2>/dev/null
  [ "${_iris_adj:-0}" -lt 0 ] 2>/dev/null && echo 0 > /proc/self/oom_score_adj 2>/dev/null
  unset _iris_adj
fi

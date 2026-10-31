# lib.sh — sourced by the test/verify scripts. sandbox_props UNIT [IRISVR_DIR]: the unit's sandbox block as
# `-p KEY=VALUE` args (one source of truth: what we test is what we ship). IRISVR_DIR = where the /data/irisvr binds come FROM (tests).
sandbox_props() {
  local unit="$1" dir="${2:-/data/irisvr}" l
  PROPS=()
  while IFS= read -r l; do
    l="${l%%#*}"; l="$(echo "$l" | xargs)"; [ -n "$l" ] || continue
    if [ "$dir" != /data/irisvr ] && [[ "$l" == Bind*Paths=/data/irisvr* ]]; then
      # tests: bind FROM the scratch dir TO the same path the bot sees (the inside view stays /data/irisvr/…)
      local k="${l%%=*}" src="${l#*=}" dst
      if [[ "$src" == *:* ]]; then dst="${src#*:}"; src="${src%%:*}"; else dst="$src"; fi
      l="$k=${src/\/data\/irisvr/$dir}:$dst"
    fi
    PROPS+=(-p "$l")
  done < <(sed -n '/--- sandbox begin/,/--- sandbox end/p' "$unit")
}
temple_host() { bash -c 'set -a; . ~iris/.iris_private.env 2>/dev/null; echo "${TEMPLE_HOST:-}"'; }
mask_ips() { sed -E 's/[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/x.x.x.x/g'; }

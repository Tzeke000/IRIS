#!/bin/sh
# :5876 bearer token (code review 10-29, security High) - see brain/orb_http.py "Bearer token".
#   orb_token.sh status            mode the NEXT runtime start will use + a fingerprint (never the token)
#   orb_token.sh mint [--rotate]   create ~/.config/iris/orb_token (0600); refuses to replace one without --rotate
#   orb_token.sh env-line FILE     append VITE_IRIS_ORB_TOKEN=<token> to FILE (0600) for the app's .env.local
# The token is never printed. Lands on the next iris-stack start (orb_http reads it at import).
# Rollout: mint -> IRIS_ORB_TOKEN_MODE=log in ~/.config/iris/stack.env -> restart -> rebuild the app with
# the token -> no "token(log): WOULD refuse" lines in state/orb_http.log for a day -> mode=enforce -> restart.
set -eu
f="${IRIS_ORB_TOKEN_FILE:-$HOME/.config/iris/orb_token}"
stack_env="${IRIS_STACK_ENV:-$HOME/.config/iris/stack.env}"
fp() { sha256sum "$f" | cut -c1-12; }
case "${1:-status}" in
  status)
    mode="$(grep -s '^IRIS_ORB_TOKEN_MODE=' "$stack_env" | tail -1 | cut -d= -f2- || true)"
    if [ -s "$f" ]; then
      echo "token: present ($f, mode $(stat -c %a "$f"), sha256 $(fp)...)"
      echo "mode:  ${mode:-log (default when a token exists)}"
    else
      echo "token: none ($f)"; echo "mode:  ${mode:-off (default without a token)}"
    fi ;;
  mint)
    if [ -s "$f" ] && [ "${2:-}" != "--rotate" ]; then
      echo "refusing: $f exists (use --rotate; the app then needs a rebuild with the new token)" >&2; exit 2
    fi
    mkdir -p "$(dirname "$f")"
    umask 077
    tmp="$f.tmp.$$"
    python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "$tmp"
    chmod 600 "$tmp"; mv -f "$tmp" "$f"
    echo "minted $f (sha256 $(fp)...)" ;;
  env-line)
    out="${2:?usage: orb_token.sh env-line FILE}"
    [ -s "$f" ] || { echo "no token at $f - mint first" >&2; exit 2; }
    umask 077; touch "$out"; chmod 600 "$out"
    printf 'VITE_IRIS_ORB_TOKEN=%s\n' "$(cat "$f")" >> "$out"
    echo "appended VITE_IRIS_ORB_TOKEN to $out (sha256 of token $(fp)...)" ;;
  *) echo "usage: orb_token.sh status|mint [--rotate]|env-line FILE" >&2; exit 2 ;;
esac

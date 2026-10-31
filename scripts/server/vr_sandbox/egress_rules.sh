#!/bin/sh
# egress_rules.sh — per-uid egress firewall for the sandboxed VR bot (audit H2).
#   egress_rules.sh install|remove UID TEMPLE_HOST [CHAIN=IRIS_VR_OUT] [CTL_PORT=7790]
# The cgroup IP filter alone allows ALL of localhost (the bot must answer on its control socket), which would let a
# hijacked bot talk to the operator API, the post-office, the mouth, the little brain… This rule set lets the bot's
# uid send exactly two things: UDP to the temple's port 7777, and replies from its own control socket (lo, tcp sport
# 7790). Everything else is REJECTED (fast EPERM / refused, so a failure is obvious, not a hang).
# Owner match only sees packets from full sockets; kernel-generated packets (SYN-ACKs, RSTs) carry no owner and are
# never caught here — the rest of the host is untouched. Installed at boot by iris-vr-egress-firewall.service.
set -eu
act="$1"; uid="$2"; temple="$3"; C="${4:-IRIS_VR_OUT}"; C6="${C}6"; CTL="${5:-7790}"
case "$act" in
install)
  iptables -N "$C" 2>/dev/null || iptables -F "$C"
  iptables -A "$C" -o lo -p tcp --sport "$CTL" -j ACCEPT
  iptables -A "$C" -d "$temple" -p udp --dport 7777 -j ACCEPT
  iptables -A "$C" -p tcp -j REJECT --reject-with tcp-reset
  iptables -A "$C" -j REJECT --reject-with icmp-admin-prohibited
  iptables -C OUTPUT -m owner --uid-owner "$uid" -j "$C" 2>/dev/null || iptables -I OUTPUT 1 -m owner --uid-owner "$uid" -j "$C"
  ip6tables -N "$C6" 2>/dev/null || ip6tables -F "$C6"
  ip6tables -A "$C6" -p tcp -j REJECT --reject-with tcp-reset
  ip6tables -A "$C6" -j REJECT --reject-with adm-prohibited
  ip6tables -C OUTPUT -m owner --uid-owner "$uid" -j "$C6" 2>/dev/null || ip6tables -I OUTPUT 1 -m owner --uid-owner "$uid" -j "$C6"
  ;;
remove)
  while iptables -D OUTPUT -m owner --uid-owner "$uid" -j "$C" 2>/dev/null; do :; done
  while ip6tables -D OUTPUT -m owner --uid-owner "$uid" -j "$C6" 2>/dev/null; do :; done
  iptables -F "$C" 2>/dev/null || true; iptables -X "$C" 2>/dev/null || true
  ip6tables -F "$C6" 2>/dev/null || true; ip6tables -X "$C6" 2>/dev/null || true
  ;;
*) echo "usage: $0 install|remove UID TEMPLE_HOST [CHAIN] [CTL_PORT]"; exit 2;;
esac

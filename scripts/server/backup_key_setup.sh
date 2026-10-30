#!/usr/bin/env bash
# scripts/server/backup_key_setup.sh - ONE-TIME: create the age key that encrypts the nightly backups.
# Zeke runs this himself, interactively (10-30): the private key is wrapped with HIS passphrase, so it can
# sit next to the backups without being able to open them. The plaintext key only ever exists in /dev/shm
# (RAM) and is shredded on exit.
#   -> ~/.config/iris/backup_age_identity.age   (passphrase-wrapped private key, server copy)
#   -> tower D:\IrisBackups\KEY\backup_age_identity.age   (second copy)
#   -> ~/.config/iris/backup_age_recipients.txt (PUBLIC key; its existence switches encryption ON)
# Restore later (age 1.0 can't use a wrapped identity directly):
#   age -d -o /dev/shm/k backup_age_identity.age   (type the passphrase)
#   restore_from_tower.sh --into DIR --latest --identity /dev/shm/k ; shred -u /dev/shm/k
set -euo pipefail
umask 077
D="$HOME/.config/iris"; WRAP="$D/backup_age_identity.age"; RECIP="$D/backup_age_recipients.txt"
mkdir -p "$D"
if [ -s "$RECIP" ] || [ -e "$WRAP" ]; then
  echo "A backup key already exists ($RECIP / $WRAP). Refusing to overwrite - old backups need the old key." >&2
  exit 1
fi
command -v age >/dev/null && command -v age-keygen >/dev/null || { echo "age not installed" >&2; exit 1; }
T=$(mktemp -d /dev/shm/agekey.XXXXXX)
trap 'shred -u "$T"/* 2>/dev/null || true; rm -rf "$T"' EXIT

age-keygen -o "$T/key" 2>/dev/null
PUB=$(age-keygen -y "$T/key")

echo
echo "== Step 1/3: choose a passphrase (you'll type it twice). Iris can NEVER recover it."
age -p -o "$WRAP.tmp" "$T/key"

echo
echo "== Step 2/3: type it ONE more time, to prove it unlocks the key."
if ! age -d -o "$T/check" "$WRAP.tmp" || ! cmp -s "$T/key" "$T/check"; then
  rm -f "$WRAP.tmp"; echo "Didn't unlock - nothing was changed. Run me again." >&2; exit 1
fi
mv "$WRAP.tmp" "$WRAP"

# round-trip with the real recipient before switching encryption on
echo "iris-backup-roundtrip" | age -r "$PUB" | age -d -i "$T/key" | grep -qx "iris-backup-roundtrip" \
  || { echo "round-trip test failed - encryption NOT switched on" >&2; exit 1; }

echo
echo "== Step 3/3: copying the wrapped key to the tower and switching encryption on"
if ssh -o BatchMode=yes -o ConnectTimeout=15 tower "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force 'D:\\IrisBackups\\KEY' | Out-Null\"" \
   && scp -q -o BatchMode=yes "$WRAP" "tower:D:/IrisBackups/KEY/backup_age_identity.age"; then
  TOWER_OK=1
else
  TOWER_OK=0
fi
printf '%s\n' "$PUB" > "$RECIP"

echo
echo "DONE. Backups are encrypted from the next run on."
echo "  wrapped key: $WRAP"
[ "$TOWER_OK" = 1 ] && echo "  tower copy:  D:\\IrisBackups\\KEY\\backup_age_identity.age" \
                     || echo "  ! tower copy FAILED - tell Iris (the server copy is fine)"
echo "  Write the passphrase down somewhere that isn't this server or the tower."

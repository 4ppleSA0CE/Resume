#!/bin/bash
# Builds the portfolio and runs every automated check.
set -u
cd "$(dirname "$0")/.."
T=portfolio_derek_wang
fail=0
latexmk -pdf -interaction=nonstopmode -halt-on-error $T.tex >/dev/null 2>&1 || { echo "FAIL build (see $T.log)"; exit 1; }
pages=$(pdfinfo $T.pdf | awk '/^Pages/{print $2}')
[ "$pages" = 5 ] || { echo "FAIL pages=$pages (want 5)"; fail=1; }
size=$(stat -f%z $T.pdf)
[ "$size" -lt 5000000 ] || { echo "FAIL size=$size bytes (want < 5000000)"; fail=1; }
if grep -v '^[[:space:]]*%' $T.tex | grep -nE -- '---|—|rather than|not only|seamless|robust|leverag|comprehensive|cutting-edge'; then echo "FAIL banned phrase"; fail=1; fi
if grep -n 'missing.png' $T.tex; then echo "FAIL placeholder image still referenced"; fail=1; fi
if grep -E 'Overfull \\hbox \(([2-9]|[0-9]{2,})\.' $T.log; then echo "FAIL overfull box > 2pt"; fail=1; fi
words=$(pdftotext $T.pdf - | wc -w)
[ "$words" -gt 800 ] || { echo "FAIL text extraction ($words words)"; fail=1; }
for u in $(grep -oE 'href\{https?://[^}]+' $T.tex | sed 's/href{//' | sort -u); do
  code=$(curl -s -o /dev/null -L -m 15 -w '%{http_code}' "$u")
  [ "$code" = 200 ] || { echo "FAIL link $code $u"; fail=1; }
done
[ $fail = 0 ] && echo "ALL CHECKS PASS"
exit $fail

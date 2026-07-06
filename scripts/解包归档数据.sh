#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ARCHIVE_DIR="$ROOT/data/归档数据"
OUT_DIR="${1:-$ROOT/data/恢复数据}"
mkdir -p "$OUT_DIR"

restore_one() {
  local dir="$1"
  local base="$2"
  local tmp
  tmp=$(mktemp "/tmp/${base}.XXXXXX.tar.gz")
  cat "$dir/${base}.tar.gz.part"* > "$tmp"
  if [ -f "$dir/${base}.tar.gz.sha256" ]; then
    (cd "$(dirname "$tmp")" && sha256sum -c <(sed "s# .*# $(basename "$tmp")#" "$dir/${base}.tar.gz.sha256"))
  fi
  tar -xzf "$tmp" -C "$OUT_DIR"
  rm -f "$tmp"
}

if [ -d "$ARCHIVE_DIR/大纲拆分" ]; then
  echo "恢复大纲拆分数据..."
  restore_one "$ARCHIVE_DIR/大纲拆分" "outline_split"
fi

if [ -d "$ARCHIVE_DIR/asr真实数据" ]; then
  for dir in "$ARCHIVE_DIR/asr真实数据"/*; do
    [ -d "$dir" ] || continue
    base=$(basename "$dir")
    echo "恢复 ASR 数据集：$base"
    restore_one "$dir" "$base"
  done
fi

echo "完成。输出目录：$OUT_DIR"

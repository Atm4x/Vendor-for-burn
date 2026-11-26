#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! -f "$1" ]]; then
  printf 'Usage: %s <linux-x64.tar.xz>\n' "$0" >&2
  exit 2
fi

archive="$(realpath "$1")"
version="tracel-llvm-23.1.0-4"
root="${XDG_DATA_HOME:-$HOME/.local/share}/tracel"
dest="$root/$version"

if [[ -x "$dest/bin/llvm-config" ]] && [[ -f "$dest/lib/libLLVMNVPTXCodeGen.a" ]] && [[ "$("$dest/bin/llvm-config" --version)" == "23.1.0" ]]; then
  touch "$dest/.tracel-llvm-installed"
  printf 'LLVM ready: %s\n' "$dest"
  printf 'LLVM_SYS_231_PREFIX=%s\n' "$dest"
  exit 0
fi

if [[ -e "$dest" ]]; then
  printf 'Refusing to overwrite existing incomplete LLVM: %s\n' "$dest" >&2
  exit 1
fi

while IFS= read -r entry; do
  if [[ "$entry" != "$version/" && "$entry" != "$version/"* ]] || [[ "$entry" == *'/../'* || "$entry" == *'/./'* ]]; then
    printf 'Unexpected archive member: %s\n' "$entry" >&2
    exit 1
  fi
done < <(tar -tJf "$archive")

mkdir -p "$root"
staging="$(mktemp -d "$root/.tracel-install.XXXXXXXX")"
trap 'rm -rf "$staging"' EXIT
tar -xJf "$archive" -C "$staging" --no-same-owner
if [[ ! -x "$staging/$version/bin/llvm-config" ]] || [[ "$("$staging/$version/bin/llvm-config" --version)" != "23.1.0" ]]; then
  printf 'Archive does not contain working Tracel LLVM 23.1.0\n' >&2
  exit 1
fi
if [[ ! -f "$staging/$version/lib/libLLVMNVPTXCodeGen.a" ]]; then
  printf 'Missing LLVM NVPTX component\n' >&2
  exit 1
fi
mv "$staging/$version" "$dest"
touch "$dest/.tracel-llvm-installed"
printf 'LLVM ready: %s\n' "$dest"
printf 'LLVM_SYS_231_PREFIX=%s\n' "$dest"

@echo off
rem Build de hayai (v0.2.3) en Windows: toolchain nightly-gnu + MinGW (msys64).
set "PATH=C:\msys64\mingw64\bin;C:\msys64\usr\bin;%PATH%"
cd /d D:\Documents\pySrc\hayai
cargo build --release

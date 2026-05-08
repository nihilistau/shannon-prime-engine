@echo off
call "D:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "D:\F\shannon-prime-repos\shannon-prime-engine\build"
ninja sp-engine 2>&1

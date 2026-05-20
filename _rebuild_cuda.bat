@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\F\shannon-prime-repos\shannon-prime-engine
cmake --build build-cuda --config Release -j 2>&1

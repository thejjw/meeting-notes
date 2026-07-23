# Build whisper.cpp for CPU on Windows
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$BuildDir = "$ProjectDir\.whisper-build"

Write-Host "=== Building whisper.cpp for CPU ===" -ForegroundColor Cyan

# Check dependencies
foreach ($cmd in @("cmake", "git")) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Host "Error: $cmd not found. Please install it." -ForegroundColor Red
        exit 1
    }
}

# Check for Visual Studio or build tools
$cmakeGenerator = $null
if (Get-Command "cl" -ErrorAction SilentlyContinue) {
    $cmakeGenerator = "Visual Studio 17 2022"
} else {
    Write-Host "Warning: Visual Studio build tools not found in PATH." -ForegroundColor Yellow
    Write-Host "You may need to run from a Visual Studio Developer Command Prompt." -ForegroundColor Yellow
}

# Clone or update whisper.cpp
if (Test-Path "$BuildDir\whisper.cpp") {
    Write-Host "Updating whisper.cpp..."
    Set-Location "$BuildDir\whisper.cpp"
    git pull
} else {
    Write-Host "Cloning whisper.cpp..."
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
    Set-Location $BuildDir
    git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git
}

Set-Location "$BuildDir\whisper.cpp"

# Build
Write-Host "Building..."
$cmakeArgs = @(
    "-B", "build",
    "-DCMAKE_BUILD_TYPE=Release"
)

if ($cmakeGenerator) {
    $cmakeArgs += "-G"
    $cmakeArgs += $cmakeGenerator
}

& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

& cmake --build build --config Release
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

Write-Host ""
Write-Host "=== Build complete ===" -ForegroundColor Green
Write-Host "Binary: $BuildDir\whisper.cpp\build\bin\Release\whisper-cli.exe"

param(
    [ValidateSet("cpu", "vulkan")]
    [string]$Backend = "cpu",
    [string]$Version = "v1.9.1"
)
$ErrorActionPreference = "Stop"

$LocalBase = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $env:USERPROFILE "AppData\Local" }
$Arch = if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq "X64") { "x86_64" } else { "arm64" }
$InstallDir = Join-Path $LocalBase "meeting-notes\cache\runtimes\$Version\windows-$Arch-$Backend"
$WorkDir = Join-Path ([System.IO.Path]::GetTempPath()) "meeting-notes-whisper-$PID"

foreach ($Command in @("cmake", "git")) {
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "$Command is required. Install CMake, Git, and Visual Studio C++ Build Tools."
    }
}
if (-not (Get-Command "cl" -ErrorAction SilentlyContinue) -and
    -not (Get-Command "clang-cl" -ErrorAction SilentlyContinue)) {
    throw "A C++ compiler is required. Run this in a Visual Studio Developer PowerShell."
}
if ($Backend -eq "vulkan" -and
    -not $env:VULKAN_SDK -and
    -not (Get-Command "vulkaninfo" -ErrorAction SilentlyContinue)) {
    throw "Vulkan SDK tooling is required for a Vulkan build."
}

try {
    git clone --branch $Version --depth 1 https://github.com/ggml-org/whisper.cpp.git "$WorkDir\source"
    $Flags = @("-S", "$WorkDir\source", "-B", "$WorkDir\build", "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=$InstallDir")
    if ($Backend -eq "vulkan") { $Flags += "-DGGML_VULKAN=1" }
    cmake @Flags
    cmake --build "$WorkDir\build" --config Release --parallel
    cmake --install "$WorkDir\build" --config Release
    & "$InstallDir\bin\whisper-cli.exe" --help | Out-Null
    Write-Host "Installed $Version $Backend runtime at $InstallDir"
} finally {
    if (Test-Path -LiteralPath $WorkDir) {
        $Resolved = (Resolve-Path -LiteralPath $WorkDir).Path
        if ($Resolved.StartsWith([System.IO.Path]::GetTempPath())) {
            Remove-Item -LiteralPath $Resolved -Recurse -Force
        }
    }
}

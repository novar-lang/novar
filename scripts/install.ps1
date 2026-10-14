# Installs `novar` and `nvpm` on Windows from a GitHub release.
# Usage:  irm https://raw.githubusercontent.com/novar-lang/novar/main/scripts/install.ps1 | iex
#         $env:NOVAR_VERSION = "0.0.3"   # optional: install a specific version
#         $env:NOVAR_LITE = "1"          # optional: ~10 MB build without SQL and Parquet
$ErrorActionPreference = "Stop"

$repo = "novar-lang/novar"
$prefix = if ($env:NOVAR_HOME) { $env:NOVAR_HOME } else { Join-Path $env:USERPROFILE ".novar" }
$bin = Join-Path $prefix "bin"
$target = "x86_64-pc-windows-msvc"

if ($env:NOVAR_VERSION) {
    $tag = "v$($env:NOVAR_VERSION)"
} else {
    # `releases/latest` skips pre-releases, which is all there is during the alpha.
    $tag = (Invoke-RestMethod "https://api.github.com/repos/$repo/releases?per_page=1")[0].tag_name
}

$asset = if ($env:NOVAR_LITE) { "novar-lite" } else { "novar" }
$url = "https://github.com/$repo/releases/download/$tag/$asset-$target.zip"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) "novar-install-$([guid]::NewGuid())"
New-Item -ItemType Directory -Force -Path $tmp, $bin | Out-Null
try {
    Invoke-WebRequest -Uri $url -OutFile (Join-Path $tmp "novar.zip")
    Expand-Archive -Path (Join-Path $tmp "novar.zip") -DestinationPath $bin -Force
} catch {
    Write-Host "no prebuilt Windows binary for $tag; build from source with: cargo install --locked --git https://github.com/$repo novar_cli novar_pkg"
    throw
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (($userPath -split ";") -notcontains $bin) {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$bin", "User")
    Write-Host "added $bin to your PATH (open a new terminal)"
}
Write-Host "installed novar $tag to $bin"

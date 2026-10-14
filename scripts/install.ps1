# Installs `telvax` and `tvpm` on Windows from a GitHub release.
# Usage:  irm https://raw.githubusercontent.com/telvax/telvax/main/scripts/install.ps1 | iex
#         $env:TELVAX_VERSION = "0.0.3"   # optional: install a specific version
#         $env:TELVAX_LITE = "1"          # optional: ~10 MB build without SQL and Parquet
$ErrorActionPreference = "Stop"

$repo = "telvax/telvax"
$prefix = if ($env:TELVAX_HOME) { $env:TELVAX_HOME } else { Join-Path $env:USERPROFILE ".telvax" }
$bin = Join-Path $prefix "bin"
$target = "x86_64-pc-windows-msvc"

if ($env:TELVAX_VERSION) {
    $tag = "v$($env:TELVAX_VERSION)"
} else {
    # `releases/latest` skips pre-releases, which is all there is during the alpha.
    $tag = (Invoke-RestMethod "https://api.github.com/repos/$repo/releases?per_page=1")[0].tag_name
}

$asset = if ($env:TELVAX_LITE) { "telvax-lite" } else { "telvax" }
$url = "https://github.com/$repo/releases/download/$tag/$asset-$target.zip"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) "telvax-install-$([guid]::NewGuid())"
New-Item -ItemType Directory -Force -Path $tmp, $bin | Out-Null
try {
    Invoke-WebRequest -Uri $url -OutFile (Join-Path $tmp "telvax.zip")
    Expand-Archive -Path (Join-Path $tmp "telvax.zip") -DestinationPath $bin -Force
} catch {
    Write-Host "no prebuilt Windows binary for $tag; build from source with: cargo install --locked --git https://github.com/$repo telvax_cli telvax_pkg"
    throw
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (($userPath -split ";") -notcontains $bin) {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$bin", "User")
    Write-Host "added $bin to your PATH (open a new terminal)"
}
Write-Host "installed telvax $tag to $bin"

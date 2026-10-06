param(
    [switch]$Sync,
    [switch]$NoFfmpeg,
    [switch]$NoUv,
    [string]$FfmpegDir = "",
    [switch]$KeepStage
)

$ErrorActionPreference = 'Stop'
$Root = if (Test-Path (Join-Path $PSScriptRoot 'pyproject.toml')) {
    $PSScriptRoot
} else {
    Split-Path -Parent $PSScriptRoot
}
Set-Location $Root

# Child tools (python/pyinstaller) print UTF-8; without this the console decodes
# their output with the ANSI code page and non-ASCII text comes out as mojibake.
try { [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false) } catch { }
$env:PYTHONIOENCODING = 'utf-8'

function Info($msg) { Write-Host "[build] $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "[build] $msg" -ForegroundColor Yellow }

# Native tools write logs to stderr; PowerShell 5.1 with ErrorActionPreference=Stop
# turns that into a terminating error. Wrap every native call and judge by exit code.
function Invoke-Native([string]$what, [scriptblock]$body) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $code = 0
    try {
        & $body 2>&1 | ForEach-Object { Write-Host $_ }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    if ($code -ne 0) { throw "$what failed (exit $code)" }
}

# ---- version / exe name ----
$pyproject = Get-Content (Join-Path $Root 'pyproject.toml') -Raw
if ($pyproject -notmatch 'version\s*=\s*"([^"]+)"') { throw "cannot read version from pyproject.toml" }
$version = $Matches[1]
$digits = ($version -replace '[^0-9]', '')
# U+526A U+6682 U+505C = "Jian Zan Ting" (the shipped exe name)
$exePrefix = [string]([char]0x526A + [char]0x6682 + [char]0x505C)
$exeName = "$exePrefix$digits"
Info "version $version -> exe name $exeName"

$py = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }

# 打包用的小脚本（内联，避免额外文件）：zip 目录并校验 UTF-8 名字标志位
$ZipCode = @'
import os, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
if os.path.isfile(dst):
    os.remove(dst)
count = 0
total = 0
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for root, dirs, files in os.walk(src):
        dirs.sort()
        files.sort()
        rel = os.path.relpath(root, src)
        if rel != ".":
            z.writestr(rel.replace(os.sep, "/") + "/", b"")
        for name in files:
            full = os.path.join(root, name)
            z.write(full, os.path.relpath(full, src).replace(os.sep, "/"))
            count += 1
            total += os.path.getsize(full)
with zipfile.ZipFile(dst) as z:
    if z.testzip():
        print("ERROR: corrupt entry in", dst)
        sys.exit(1)
    for info in z.infolist():
        if not info.filename.isascii() and not (info.flag_bits & 0x800):
            print("ERROR: missing UTF-8 name flag for", info.filename)
            sys.exit(1)
print("zip entries=%d raw=%.1fMB out=%.1fMB" % (
    count, total / 1048576, os.path.getsize(dst) / 1048576))
'@

# ---- dependency sync (off by default) ----
if ($Sync) {
    Info "uv sync (this PRUNES packages outside the lockfile) ..."
    Invoke-Native 'uv sync' { & uv sync }
} else {
    Info "skipping uv sync (pass -Sync to enable)"
}

# ---- PyInstaller ----
Info "running PyInstaller (first run is slow) ..."
$env:AAE_EXE_NAME = $exeName
$specPath = Join-Path $Root 'arknight.spec'
$distPath = Join-Path $Root 'dist'
$workPath = Join-Path $Root 'build'
$pyi = Join-Path $Root '.venv\Scripts\pyinstaller.exe'
if (Test-Path $pyi) {
    Invoke-Native 'PyInstaller' {
        & $pyi --noconfirm --clean --log-level WARN `
            --distpath $distPath --workpath $workPath $specPath
    }
} else {
    Warn "pyinstaller not found in .venv, falling back to 'uv run pyinstaller'"
    Invoke-Native 'PyInstaller(uv run)' {
        & uv run pyinstaller --noconfirm --clean --log-level WARN `
            --distpath $distPath --workpath $workPath $specPath
    }
}

$exePath = Join-Path $distPath "$exeName.exe"
if (-not (Test-Path $exePath)) { throw "missing artifact: $exePath" }
Info ("exe size {0:N1} MB" -f ((Get-Item $exePath).Length / 1MB))

# ---- staging ----
$DATA_DIRS = @('templates_pause','templates_1x','templates_2x','templates_play',
               'source_images_pause','source_images_1x','source_images_2x','source_images_play')

function New-Stage([string]$name) {
    $p = Join-Path $Root "dist\$name"
    if (Test-Path $p) { Remove-Item -Recurse -Force $p }
    New-Item -ItemType Directory -Path $p | Out-Null
    Copy-Item $exePath $p
    Copy-Item (Join-Path $Root 'README.md') $p -ErrorAction SilentlyContinue
    foreach ($d in $DATA_DIRS) {
        $src = Join-Path $Root $d
        if (Test-Path $src) { Copy-Item -Recurse $src (Join-Path $p $d) }
        else { Warn "missing asset dir: $d" }
    }
    return $p
}

function New-Zip([string]$stage, [string]$zip) {
    if (Test-Path $zip) { Remove-Item -Force $zip }
    # 用 Python 的 zipfile：非 ASCII 条目名会带上 UTF-8 标志位。
    # .NET 的 CreateFromDirectory 传 entryNameEncoding 时不置该标志位，中文 exe 名
    # 在解压端会变乱码（Explorer 直接报 Illegal characters in path）。
    # 代码走临时文件而不是 python -c：PowerShell 5.1 传原生参数时会吃掉内嵌引号。
    $tmpPy = Join-Path $env:TEMP ("aae_zip_" + [guid]::NewGuid().ToString("N") + ".py")
    try {
        Set-Content -Path $tmpPy -Value $ZipCode -Encoding UTF8
        Invoke-Native 'zip' {
            & $py $tmpPy $stage $zip
        }
    } finally {
        Remove-Item $tmpPy -Force -ErrorAction SilentlyContinue
    }
    if (-not (Test-Path $zip)) { throw "zip not produced: $zip" }
    Info ("zip {0} ({1:N1} MB)" -f (Split-Path -Leaf $zip), ((Get-Item $zip).Length / 1MB))
}

# ---- main pack (reference-shaped) ----
$liteStage = New-Stage "stage-$digits"
New-Zip $liteStage (Join-Path $distPath "v$version-win.zip")

# ---- full pack (main + ffmpeg + uv) ----
$wantFull = (-not $NoFfmpeg) -or (-not $NoUv)
if ($wantFull) {
    $fullStage = New-Stage "stage-$digits-full"
    $added = @()

    if (-not $NoFfmpeg) {
        $ffDir = $FfmpegDir
        if (-not $ffDir) {
            $cmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
            if ($cmd) { $ffDir = Split-Path -Parent $cmd.Source }
        }
        if ($ffDir -and (Test-Path (Join-Path $ffDir 'ffmpeg.exe'))) {
            Copy-Item (Join-Path $ffDir 'ffmpeg.exe') $fullStage
            if (Test-Path (Join-Path $ffDir 'ffprobe.exe')) {
                Copy-Item (Join-Path $ffDir 'ffprobe.exe') $fullStage
            }
            Get-ChildItem -Path $ffDir -Filter *.dll -ErrorAction SilentlyContinue |
                ForEach-Object { Copy-Item $_.FullName $fullStage }
            $added += 'ffmpeg'
        } else {
            Warn "ffmpeg.exe not found: full pack will not bundle ffmpeg"
        }
    }

    if (-not $NoUv) {
        $uv = Get-Command uv -ErrorAction SilentlyContinue
        if ($uv) {
            Copy-Item $uv.Source (Join-Path $fullStage 'uv.exe') -ErrorAction SilentlyContinue
            $added += 'uv'
        } else {
            Warn "uv not found: in-app CUDA pack download will be unavailable"
        }
    }

    if ($added.Count -gt 0) {
        New-Zip $fullStage (Join-Path $distPath "v$version-win-full.zip")
        Info ("full pack = main + " + ($added -join ' + '))
    } else {
        Warn "no optional component available, skipping full pack"
    }
    if (-not $KeepStage) { Remove-Item -Recurse -Force $fullStage -ErrorAction SilentlyContinue }
}

if (-not $KeepStage) { Remove-Item -Recurse -Force $liteStage -ErrorAction SilentlyContinue }

Write-Host ""
Info "dist:"
Get-ChildItem $distPath -File | ForEach-Object {
    Write-Host ("   {0}  {1:N1} MB" -f $_.Name, ($_.Length / 1MB))
}
Info "done"

param(
    [string]$CudaIndex = "https://download.pytorch.org/whl/cu128",
    [string]$PythonVersion = "",
    [string]$Platform = "windows",
    [switch]$SkipZip
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Info($msg) { Write-Host "[gpu-pack] $msg" -ForegroundColor Cyan }

# meow ~: pack for THIS interpreter instead of a hard-coded 3.13.
# whoever runs the app must share the pack's CPython minor version (torch C ext ABI).
if (-not $PythonVersion) {
    $venvPy = Join-Path $Root '.venv\Scripts\python.exe'
    if (Test-Path $venvPy) {
        $PythonVersion = & $venvPy -c "import sys;print('%d.%d' % sys.version_info[:2])"
    }
}
if (-not $PythonVersion) {
    $pyLauncher = (Get-Command py -ErrorAction SilentlyContinue).Source
    if ($pyLauncher) {
        $PythonVersion = & $pyLauncher -3 -c "import sys;print('%d.%d' % sys.version_info[:2])"
    }
}
if (-not $PythonVersion) {
    throw "cannot detect Python version; pass -PythonVersion 3.12 (the version that will load this pack)"
}
$PythonVersion = "$PythonVersion".Trim()
Info "python version: $PythonVersion"

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

$pyproject = Get-Content (Join-Path $Root 'pyproject.toml') -Raw
if ($pyproject -notmatch 'version\s*=\s*"([^"]+)"') { throw "cannot read version" }
$version = $Matches[1]

$outDir = Join-Path $Root 'packaging\out'
$target = Join-Path $outDir 'gpu-torch'
if (Test-Path $target) { Remove-Item -Recurse -Force $target }
New-Item -ItemType Directory -Path $target -Force | Out-Null

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { throw "uv not found; install uv first" }

Info "downloading torch (CUDA) into $target ..."
Invoke-Native 'uv pip install' {
    & $uv pip install `
        --target $target `
        --python-platform $Platform `
        --python-version $PythonVersion `
        --index-url $CudaIndex `
        --upgrade torch
}

if (-not (Test-Path (Join-Path $target 'torch'))) { throw "no torch directory in result" }

# meow ~: read the ABI out of _C.<tag>.pyd, that is what the app checks before loading
$abiFile = Get-ChildItem (Join-Path $target 'torch') -Filter '_C.*.pyd' -ErrorAction SilentlyContinue |
           Select-Object -First 1
$abiTag = if ($abiFile) { $abiFile.Name -replace '^_C\.', '' -replace '\.pyd$', '' } else { '' }
if ($abiTag) {
    Info "python abi: $abiTag (only a matching CPython can load this pack)"
} else {
    Info "python abi: unknown (no _C.<tag>.pyd found)"
}

$py = Join-Path $Root '.venv\Scripts\python.exe'
if (Test-Path $py) {
    $torchVersion = & $py -c "import sys;sys.path.insert(0,r'$target');import torch;print(torch.__version__)" 2>$null
} else {
    $torchVersion = 'unknown'
}
Info "torch version: $torchVersion"

$size = (Get-ChildItem $target -Recurse -File | Measure-Object -Property Length -Sum).Sum
Info ("pack size {0:N1} MB" -f ($size / 1MB))

if ($SkipZip) { Info "skipped zipping (-SkipZip)"; exit 0 }

$zip = Join-Path $outDir "v$version-gpu-torch.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }

Info "zipping (2~3GB, slow) ..."
if (-not (Test-Path $py)) { $py = 'python' }
Invoke-Native 'zip' {
    & $py (Join-Path $Root 'packaging\make_zip.py') $target $zip
}

$hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
$unpackTo = "%LOCALAPPDATA%\arknight-auto-editing\gpu-torch"
if ($abiTag) { $unpackTo = "$unpackTo\$abiTag" }
$manifest = [ordered]@{
    name          = "arknight-auto-editing gpu-torch pack"
    app_version   = $version
    torch_version = "$torchVersion".Trim()
    cuda_index    = $CudaIndex
    platform      = $Platform
    python        = $PythonVersion
    python_abi    = $abiTag
    zip           = (Split-Path -Leaf $zip)
    zip_bytes     = (Get-Item $zip).Length
    sha256        = $hash
    unpack_to     = $unpackTo
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $outDir 'manifest.json') -Encoding UTF8

Info ("pack {0} ({1:N1} MB)" -f $zip, ((Get-Item $zip).Length / 1MB))
Info "sha256 = $hash"
Info "host the zip and set ARKNIGHT_GPU_PACK_URL (or config gpu_pack_url) for in-app download"

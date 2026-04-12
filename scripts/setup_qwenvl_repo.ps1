param(
    [string]$RepoDir = "third_party/Qwen2.5-VL",
    [string]$RepoUrl = "https://github.com/QwenLM/Qwen2.5-VL.git",
    [switch]$Pull
)

# 中文说明：
# - 默认行为：如果仓库不存在则 clone。
# - 若仓库已存在，只有传入 -Pull 才执行 git pull。
# - 用法示例：
#   powershell ./scripts/setup_qwenvl_repo.ps1
#   powershell ./scripts/setup_qwenvl_repo.ps1 -Pull

$ErrorActionPreference = "Stop"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is not installed or not in PATH（未检测到 git，可先安装 Git for Windows）"
}

if (-not (Test-Path $RepoDir)) {
    Write-Host "Cloning Qwen2.5-VL repo to $RepoDir ..."
    git clone $RepoUrl $RepoDir
} elseif ($Pull) {
    Write-Host "Updating existing repo at $RepoDir ..."
    Push-Location $RepoDir
    git pull
    Pop-Location
} else {
    Write-Host "Repo already exists: $RepoDir（如需更新请加 -Pull）"
}

Write-Host "Done."

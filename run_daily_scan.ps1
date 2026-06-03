# run_daily_scan.ps1 — 由 Windows 工作排程器呼叫
# 每天 18:00 執行（T86 法人資料約 17:30 發布）

$env:GMAIL_APP_PW = (Get-Content "$PSScriptRoot\.gmail_app_pw" -Raw).Trim()

$logFile = "$PSScriptRoot\scan_log.txt"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

Add-Content $logFile "`n========== $timestamp =========="
& python "$PSScriptRoot\daily_scan.py" 2>&1 | Tee-Object -Append $logFile

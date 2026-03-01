# Script để cài đặt Python và pip trên Windows
Write-Host "Đang kiểm tra và cài đặt Python..." -ForegroundColor Cyan

# Kiểm tra xem Python đã được cài đặt chưa
$pythonInstalled = $false
$pythonPaths = @(
    "C:\Program Files\Python*",
    "$env:LOCALAPPDATA\Programs\Python",
    "$env:ProgramFiles\Python*"
)

foreach ($path in $pythonPaths) {
    if (Test-Path $path) {
        $pythonInstalled = $true
        Write-Host "Đã tìm thấy Python tại: $path" -ForegroundColor Green
        break
    }
}

if (-not $pythonInstalled) {
    Write-Host "Python chưa được cài đặt." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Vui lòng cài đặt Python theo một trong các cách sau:" -ForegroundColor Cyan
    Write-Host "1. Tải từ python.org: https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "   - Chọn Python 3.11 hoặc mới hơn" -ForegroundColor Gray
    Write-Host "   - Khi cài đặt, NHỚ CHỌN 'Add Python to PATH'" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "2. Hoặc cài qua Microsoft Store:" -ForegroundColor White
    Write-Host "   - Mở Microsoft Store" -ForegroundColor Gray
    Write-Host "   - Tìm 'Python 3.11' hoặc 'Python 3.12'" -ForegroundColor Gray
    Write-Host "   - Cài đặt" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. Hoặc cài qua winget (nếu có):" -ForegroundColor White
    Write-Host "   winget install Python.Python.3.11" -ForegroundColor Gray
    Write-Host ""
    
    # Hỏi xem có muốn cài qua winget không
    $useWinget = Read-Host "Bạn có muốn thử cài đặt qua winget? (y/n)"
    if ($useWinget -eq 'y' -or $useWinget -eq 'Y') {
        try {
            Write-Host "Đang cài đặt Python qua winget..." -ForegroundColor Cyan
            winget install Python.Python.3.11 --accept-package-agreements --accept-source-agreements
            Write-Host "Cài đặt hoàn tất! Vui lòng đóng và mở lại terminal." -ForegroundColor Green
        } catch {
            Write-Host "Không thể cài đặt qua winget. Vui lòng cài đặt thủ công." -ForegroundColor Red
        }
    }
} else {
    Write-Host "Python đã được cài đặt!" -ForegroundColor Green
}

Write-Host ""
Write-Host "Sau khi cài đặt Python, chạy lệnh sau để cài đặt dependencies:" -ForegroundColor Cyan
Write-Host "python -m pip install -r requirements.txt" -ForegroundColor Yellow
Write-Host ""
Write-Host "Hoặc nếu dùng py launcher:" -ForegroundColor Cyan
Write-Host "py -m pip install -r requirements.txt" -ForegroundColor Yellow

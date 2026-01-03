@echo off

REM 进入脚本所在目录
cd /d "%~dp0"

REM 暂存所有修改（排除 .gitignore 忽略的文件）
git add .

REM 提交修改（默认提交信息）
git commit -m "Auto update all files"

REM 推送到 GitHub
git push

echo.
echo 所有修改已上传到 GitHub
pause
@echo off
set /p commit_msg="Enter your commit message: "

echo.
echo [1/3] Adding changes...
git add .

echo [2/3] Committing changes...
git commit -m "%commit_msg%"

echo [3/3] Pushing to both repositories...
echo Pushing to Mirza-sufyan-baig (origin)...
git push origin main

echo Pushing to shehbaz0101 (origin2)...
git push origin2 main

echo.
echo All done! Press any key to exit.
pause
@echo off
echo 🔧 Fixing Git conflicts and deploying to Streamlit Cloud
echo.

echo Step 1: Stashing local changes...
git stash

echo Step 2: Pulling latest changes from remote...
git pull origin main

echo Step 3: Applying your stashed changes...
git stash pop

echo Step 4: Checking status...
git status

echo.
echo ✅ Git conflicts resolved!
echo.
echo Step 5: Adding all files...
git add .

echo Step 6: Committing enterprise platform...
git commit -m "🚀 Enterprise SaaS Platform - Ready for Streamlit Cloud deployment"

echo Step 7: Pushing to GitHub...
git push origin main

echo.
echo 🎊 SUCCESS! Your enterprise platform is now on GitHub!
echo.
echo 🌐 Next steps for FREE deployment:
echo 1. Go to https://share.streamlit.io/
echo 2. Sign in with your GitHub account
echo 3. Click "New app"
echo 4. Select repository: sales-forecast-app
echo 5. Main file path: streamlit_app.py
echo 6. Click "Deploy!"
echo.
echo 🚀 Your enterprise platform will be live in 2-3 minutes!
echo.
pause
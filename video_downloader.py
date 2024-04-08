import os

from supervision.assets import VideoAssets, download_assets

if not os.path.exists("data"):
    os.markedirs("data")
os.chdir("data")

download_assets(VideoAssets.VEHICLES)

print("Files are saved in:", os.getcwd())
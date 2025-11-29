from src.youtube_downloader import YouTubeDownloader

if __name__ == "__main__": 
    artists = ["Clairo", "Rush", "Juice WRLD"]
    downloader = YouTubeDownloader()
    downloader.download_artists(artists, max_per_artist=15)
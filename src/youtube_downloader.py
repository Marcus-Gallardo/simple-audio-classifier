import yt_dlp
import json
import os

from src.utils import get_project_root, get_raw_audio_path, sanitize_for_path
from ytmusicapi import YTMusic

class YouTubeDownloader():

    songs_dict = {}

    def __init__(self):
        self.songs_dict = self._get_downloaded_songs()

        self.yt = YTMusic()

    # Gets the already-downloaded songs from the raw audio directory
    def _get_downloaded_songs(self):

        # Get path to raw audio directory
        raw_audio_path = get_raw_audio_path()

        downloaded_songs = {}

        # Go through each artist directory and list .wav files
        for dirname in os.listdir(raw_audio_path):

            # Get artist directory path
            artist_dir = os.path.join(raw_audio_path, dirname)

            # Continue if not the artist directory
            if not os.path.isdir(artist_dir):
                continue

            # Read metadata to get real artist name
            meta = self.read_artist_metadata(artist_dir)

            if not meta:
                print(f"{artist_dir} has no metadata. Skipping")
                continue

            # Get official artist name
            official_artist_name = meta.get("artist", dirname) # Fall back to dirname if get fails
            
            # Get songs
            songs = meta.get("songs", {})
            
            # Save songs to dict
            downloaded_songs[official_artist_name] = list(songs.keys())

        return downloaded_songs

    def _song_already_downloaded(self, artist, song_title):
        if artist in self.songs_dict:
            if song_title in self.songs_dict[artist]:
                print(f"Song '{song_title}' by '{artist}' already downloaded. Skipping.")
                return True
        
        print(f"Downloading song '{song_title}' by '{artist}'.")
        return False

    # Downloads audio from a YouTube URL and saves it as a .wav file
    def _download_youtube_audio(self, url, artist_dir, song_title):
        # Check if artist directory exists
        if not os.path.exists(artist_dir):
            print(f"Could not find artist directory \"{artist_dir}\" for downloading audio.")
            return

        # Download into artist directory
        output_template = os.path.join(artist_dir, "%(title)s.%(ext)s") # The percent signs are yt-dlp syntax

        # Configure yt-dlp download
        ytdlp_ops = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "wav"},
            ],
        }

        with yt_dlp.YoutubeDL(ytdlp_ops) as ydl:
            info = ydl.extract_info(url, download=True)

        # Rename file with normalized filename
        if info:
            # Get path to downloaded file
            downloaded_path = info["requested_downloads"][0]["filepath"]

            # Split into filename and extension
            _, ext = os.path.splitext(downloaded_path)

            final_path = os.path.join(artist_dir, sanitize_for_path(song_title) + ext)
    
            # Rename the file if necessary
            if downloaded_path != final_path:
                print("Renaming downloaded file to normalized filename.")
                os.rename(downloaded_path, final_path)
        else:
            print("Could not find info about downloaded file.")

    # Downloads the top songs of a given artist
    def download_artist_audio(self, artist_name, max_per_artist=3):

        # Search YouTube Music for the artist
        artist_results = self.yt.search(artist_name, filter="artists")

        if not artist_results:
            print(f"No results for '{artist_name}' on YouTube Music. Skipping.")
            return []

        # Get the official artist name
        official_artist_name = artist_results[0]["artist"]

        if not official_artist_name:
            print(f"Could not find an official name for '{artist_name}' on YouTube Music. Skipping.")
            return
        else:
            print(f"Found artist '{official_artist_name}' for search '{artist_name}'.")

        # Get the first artist result
        artist_id = artist_results[0]["browseId"]

        # Get artist's official songs
        song_results = self.yt.search(artist_name, filter="songs")
        
        # Filter only songs where the first listed artist matches exactly
        songs = [
            r for r in song_results
            if r.get("artists") and r["artists"][0].get("id") == artist_id
        ]

        if not songs:
            print(f"No songs found for artist '{official_artist_name}'. Skipping.")
            return
    
        # Create artist directory and write metadata
        root = get_project_root()
        artist_dir = os.path.join(root, "data", "raw_audio", sanitize_for_path(official_artist_name))
        os.makedirs(artist_dir, exist_ok=True)
        self.write_artist_metadata(artist_dir, official_artist_name, artist_id)

        # Download the top songs up to the max per artist
        for song in songs[:max_per_artist]:

            # Don't download more songs than the max number asked by the user
            if self.get_num_songs_by_artist(official_artist_name) > max_per_artist:
                break

            # Don't download the song if it's already been downloaded
            if self._song_already_downloaded(official_artist_name, song["title"]):
                continue

            video_id = song["videoId"]
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            self._download_youtube_audio(youtube_url, artist_dir, song['title'])
            self.add_song_to_metadata(artist_dir, song["title"], f"{sanitize_for_path((song['title']))}.wav")

    # Downloads songs based on the given song names
    def download_songs(self, songs):
        for song_descriptor in songs:
            # Search YouTube Music for the artist
            song_results = self.yt.search(song_descriptor, filter="songs")

            if not song_results:
                print(f"No results for '{song_descriptor}' on YouTube Music. Skipping.")
                return []

            # Get the artist name (first artist)
            official_artist_name = song_results[0]["artists"][0]["name"]
            
            if not official_artist_name:
                print(f"Could not find the artist for '{song_descriptor}' on YouTube Music. Skipping.")
                return
            else:
                print(f"Found artist '{official_artist_name}' for '{song_descriptor}'.")

            # Get the artist's ID
            artist_id = song_results[0]["artists"][0]["id"]

            # Create artist directory and write metadata
            root = get_project_root()
            artist_dir = os.path.join(root, "data", "raw_audio", sanitize_for_path(official_artist_name))
            os.makedirs(artist_dir, exist_ok=True)
            self.write_artist_metadata(artist_dir, official_artist_name, artist_id)

            song_title = song_results[0]["title"]

            # Don't download the song if it's already been downloaded
            if self._song_already_downloaded(official_artist_name, song_title):
                continue

            video_id = song_results[0]["videoId"]
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            self._download_youtube_audio(youtube_url, artist_dir, song_title)
            self.add_song_to_metadata(artist_dir, song_title, f"{sanitize_for_path((song_title))}.wav")

    # Returns the number of songs in the dataset.
    def get_num_songs(self):
        total_songs = 0
        for songs in self.songs_dict.values():
            total_songs += len(songs)
        return total_songs
    
    # Returns the number of songs in the dataset by the given artist
    def get_num_songs_by_artist(self, artist):
        if artist in self.songs_dict:
            return len(self.songs_dict[artist])
        else:
            return 0

    # Downloads the songs for the given artists
    def download_artists(self, artists, max_per_artist=3):
        for artist in artists:
            self.download_artist_audio(artist, max_per_artist=max_per_artist)

    def write_artist_metadata(self, artist_dir, artist_name, artist_id):

        # Define metadata path        
        meta_path = os.path.join(artist_dir, "metadata.json")

        # Read existing metadata (returns None if missing)
        existing = self.read_artist_metadata(artist_dir)

        # If metadata does not exist, start fresh
        if existing is None:
            existing = {}

        # Construct new metadata base
        meta = {"artist": artist_name}
        if artist_id:
            meta["artist_id"] = artist_id

        # Merge existing songs
        meta["songs"] = existing.get("songs", {})

        # Preserve other keys
        for k, v in existing.items():
            if k not in meta:
                meta[k] = v

        # Write metadata file
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def read_artist_metadata(self, artist_dir):

        meta_path = os.path.join(artist_dir, "metadata.json")

        if not os.path.exists(meta_path):
            return None

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        return meta

    def add_song_to_metadata(self, artist_dir, song_name, filename):

        # Read existing metadata
        meta = self.read_artist_metadata(artist_dir)
        if meta is None:
            print(f"No metadata found in {artist_dir}. Cannot add song.")
            return

        # Update songs
        songs = meta.get("songs", {})
        songs[song_name] = filename
        meta["songs"] = songs

        # Write update
        meta_path = os.path.join(artist_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

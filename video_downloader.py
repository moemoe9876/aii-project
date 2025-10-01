#!/usr/bin/env python3
"""
Video Downloader Script
Downloads videos from Instagram, Twitter, and other supported platforms.
"""

import sys
import os
import shutil
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    print("Error: yt-dlp is not installed.")
    print("Please install it using: pip install yt-dlp")
    sys.exit(1)


class VideoDownloader:
    def __init__(self, output_dir="downloads"):
        """
        Initialize the video downloader.
        
        Args:
            output_dir: Directory where videos will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        # Optional cookies support (helps with sites like X/Twitter)
        self.cookies_browser = os.getenv('YTDLP_COOKIES_FROM_BROWSER')  # e.g., 'chrome', 'firefox'
        self.cookies_profile = os.getenv('YTDLP_COOKIES_PROFILE')       # e.g., 'Default'
        self.cookies_file = os.getenv('YTDLP_COOKIES_FILE')             # path to Netscape cookies.txt
        
    def _ffmpeg_available(self) -> bool:
        """Return True if ffmpeg is available on PATH."""
        return shutil.which("ffmpeg") is not None

    def download_video(self, url):
        """
        Download a video from the given URL with original audio.
        
        Args:
            url: The URL of the video to download
            
        Returns:
            bool: True if download was successful, False otherwise
        """
        # Check if ffmpeg is available for merging video+audio streams
        ffmpeg_ok = self._ffmpeg_available()
        if not ffmpeg_ok:
            print("[!] ffmpeg not found on PATH. Falling back to progressive formats.")
            print("    Install ffmpeg to merge separate video+audio for best quality.")

        # Format selection ensuring audio is ALWAYS included:
        # With ffmpeg: Download best video + best audio separately, then merge
        # Without ffmpeg: Download best single file that contains both video and audio
        #
        # Format string breakdown:
        # - bestvideo[ext=mp4]+bestaudio[ext=m4a]: Best MP4 video + best M4A audio (merged)
        # - bestvideo+bestaudio: Best video + best audio in any format (merged)
        # - best[height<=?1080]: Best single format up to 1080p (with audio)
        # - best: Absolute fallback - best available format
        fmt_best_merge = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[height<=?1080]/best'
        
        # Progressive format: Single file with both video and audio (no merging needed)
        # - [acodec!=none]: Ensures audio codec is present (not a silent video)
        fmt_progressive = 'best[ext=mp4][acodec!=none]/best[height<=?1080][acodec!=none]/best'

        ydl_opts = {
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'format': fmt_best_merge if ffmpeg_ok else fmt_progressive,
            'prefer_ffmpeg': True,
            'noplaylist': True,
            'quiet': False,
            'no_warnings': False,
            'ignoreerrors': True,
            'keepvideo': False,
            # Add a common User-Agent in case some CDNs are picky
            'http_headers': {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'},
        }
        # Attach cookies if configured
        if self.cookies_browser:
            if self.cookies_profile:
                ydl_opts['cookiesfrombrowser'] = (self.cookies_browser, self.cookies_profile)
            else:
                ydl_opts['cookiesfrombrowser'] = (self.cookies_browser,)
        if self.cookies_file:
            ydl_opts['cookiefile'] = self.cookies_file
        
        # Configure ffmpeg post-processing to ensure audio is properly embedded
        if ffmpeg_ok:
            ydl_opts['merge_output_format'] = 'mp4'
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }]
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print(f"\n[+] Downloading from: {url}")
                info = ydl.extract_info(url, download=True)
                if info:
                    print(f"[✓] Successfully downloaded: {info.get('title', 'Unknown')}")
                    return True
                else:
                    print(f"[✗] Failed to download from: {url}")
                    return False
        except Exception as e:
            print(f"[✗] Error downloading {url}: {str(e)}")
            return False
    
    def download_multiple(self, urls):
        """
        Download multiple videos from a list of URLs.
        
        Args:
            urls: List of URLs to download
            
        Returns:
            tuple: (successful_count, failed_count)
        """
        successful = 0
        failed = 0
        
        for url in urls:
            if self.download_video(url):
                successful += 1
            else:
                failed += 1
        
        return successful, failed


def main():
    """Main function to handle command-line usage."""
    print("=" * 60)
    print("Video Downloader - Instagram, Twitter & More")
    print("Downloads videos WITH ORIGINAL AUDIO")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  Single URL:    python video_downloader.py <URL>")
        print("  Multiple URLs: python video_downloader.py <URL1> <URL2> <URL3>...")
        print("  From file:     python video_downloader.py --file urls.txt")
        print("\nExamples:")
        print("  python video_downloader.py https://www.instagram.com/p/...")
        print("  python video_downloader.py https://twitter.com/user/status/...")
        print("  python video_downloader.py --file urls.txt")
        print("\nNote:")
        print("  • All videos are downloaded with original audio")
        print("  • ffmpeg is recommended for best quality (merges separate video+audio)")
        print("  • Videos will be saved to the 'downloads' folder")
        sys.exit(1)
    
    downloader = VideoDownloader()
    urls = []
    
    if sys.argv[1] == "--file":
        if len(sys.argv) < 3:
            print("[✗] Error: Please specify the file path")
            sys.exit(1)
        
        file_path = sys.argv[2]
        try:
            with open(file_path, 'r') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        except FileNotFoundError:
            print(f"[✗] Error: File '{file_path}' not found")
            sys.exit(1)
    else:
        urls = sys.argv[1:]
    
    if not urls:
        print("[✗] No URLs provided")
        sys.exit(1)
    
    print(f"\n[*] Found {len(urls)} URL(s) to download")
    print(f"[*] Output directory: {downloader.output_dir.absolute()}\n")
    
    if len(urls) == 1:
        success = downloader.download_video(urls[0])
        if success:
            print("\n[✓] Download completed!")
        else:
            print("\n[✗] Download failed!")
            sys.exit(1)
    else:
        successful, failed = downloader.download_multiple(urls)
        print("\n" + "=" * 60)
        print(f"Download Summary:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total: {len(urls)}")
        print("=" * 60)


if __name__ == "__main__":
    main()

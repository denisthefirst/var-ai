import os
import re
import yaml
import yt_dlp
from yt_dlp.utils import download_range_func

script_dir = os.path.dirname(__file__)
proj_root_dir = os.path.join(script_dir, '..', '..')
video_collection_path = os.path.join(proj_root_dir, 'video-collection.yaml')
video_download_path = os.path.join(proj_root_dir, 'data', 'raw')
titles: dict = {}
downloaded_files: list = []


def scan_download_dir() -> list:
    """
    This function returns all filenames inside the download
    directory.

    :returns: A list with filenames and file extension.
    """
    try:
        return os.listdir(video_download_path)
    except Exception as e:
        print(f"Error reading download directory: {e}")
        return []


def file_already_exists(filename: str) -> bool:
    """
    Check wehter given file already exists.

    :param filename: The filename to check for.
    :returns: Boolean value based on avaliablity of given file.
    """
    if filename in downloaded_files:
        return True
    else:
        return False


def extract_url_and_start(url_start: str) -> tuple:
    """
    Extract the video URL and starting time stamp from a
    given URL.
        Example: 
        `https://www.youtube.com/watch?v=f5M9mzcjZ_8&t=7890s` is the 
        `url_start` variable, here `7890` is the starting timestamp, and
        `https://www.youtube.com/watch?v=f5M9mzcjZ_8` the URL of the video

    :param url_start: The combination of video url and timestamp.
    :returns touple: A touple with the video URL and its starting time.
    """
    try:
        match_start_time = re.search(r"&t=(\d+(?:\.\d+)?)s", url_start)
        if match_start_time:
            start_time = round(float(match_start_time.group(1)), 2)
        url = re.sub(r"&t=\d+s", "", url_start)
        return (f"{url}", start_time)
    except Exception as e:
        print(f"Failed to extract URL and start time from {url_start}: {e}")
        return (url_start, 0.0)


def download_video(video_title: str,
                   video_url: str,
                   section_title: str,
                   start_time: float,
                   end_time: float) -> None:
    """
    This function downloads a video from the specified URL.
    It generates filenames based on the video and section title.
    Videos are stored inside the globally given video download path.
    If a video with the generated filename already exists, it is not
    downloaded.
    The given timestamps provide the section of those videos. Only sections
    are downloaded, not the whole video. This saves on resources.

    :param video_title: The main title of the video.
    :param section_title: The section title, this is a section inside the main video.
    :param start_time: The starting time stamp.
    :param end_time: The ending time stamp.
    """

    # Add a counter to the section title if the same section title
    # already exists, this prevents overriding files
    if f"{video_title}-{section_title}" in titles:
        titles[f"{video_title}-{section_title}"] += 1
        suffix = titles[f"{video_title}-{section_title}"]
        section_title = f"{section_title}-{suffix}"
    else:
        titles[f"{video_title}-{section_title}"] = 0

    filename: str = f"{video_title}-{section_title}"

    if not file_already_exists(f"{filename}.mp4"):
        output = os.path.join(video_download_path, filename)

        ydl_opts: dict = {
            'verbose': False,
            'quiet': True,
            'format': 'bv*[ext=mp4][vcodec^=avc]/bv*',
            'download_ranges': download_range_func(None, [(start_time, end_time)]),
            'merge-output-format': 'mp4',
            'remux-video': 'mp4',
            'no-audio': True,
            'outtmpl': f"{output}.mp4"
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([video_url])
            except Exception as e:
                print(f"Failed to download video \"{filename}\" from {video_url}: {e}")


def collect_videos() -> None:
    """
    This function straps everything together.
    It checks already collected videos, reads in the video collection 
    `.yaml`-file and then calls the function to download the videos
    with specified parameters.
    """

    global downloaded_files
    # Write all filenames from the download directory to a list,
    # This list is later used to determine if a video has already been
    # downloaded. This allows adding new videos to the collection
    # without having to download them again.
    downloaded_files = scan_download_dir()
   
    try:
        with open(video_collection_path, "r") as file:
            video_collection: dict = yaml.safe_load(file)
    except Exception as e:
        print(f"Failed to load video collection YAML: {e}")
        return
    
    for video in video_collection:
        video_title: str = video["title"]
        print(f"Video with title: \"{video_title}\", contains the following sections:")
        for section in video["sections"]:
            section_title: str = section["title"]
            url_start: tuple = extract_url_and_start(section["url"])
            url: str = url_start[0]
            start_time: float = url_start[1]
            end_time: float = start_time + round(float(section["duration"]), 2)
            print(f"\tSection: \"{section_title}\"")
            print(f"\t\turl_start: \"{url_start}\"")
            print(f"\t\tend_time: \"{end_time}\"")
            download_video(video_title, url, section_title, start_time, end_time)
        print("\n")


def main():
    collect_videos()


if __name__ == "__main__":
    main()

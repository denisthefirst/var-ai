import os
import yaml
import yt_dlp
from yt_dlp.utils import download_range_func

script_dir = os.path.dirname(__file__)
proj_root_dir = os.path.join(script_dir, '..', '..')
video_collection_path = os.path.join(proj_root_dir, 'video-collection.yaml')
video_download_path = os.path.join(proj_root_dir, 'data', 'videos')
titles: dict = {}
downloaded_files: list = []


def get_all_files(path) -> list:
    """
    This function returns all filenames inside the download
    directory.

    :returns: A list with filenames and file extension.
    """
    all_files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            full_path = os.path.join(dirpath, file)
            all_files.append(full_path)
    return all_files


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


def timestamp_to_seconds(timestamp: str) -> int:
    """
    Convert a timestamp "hh:mm:ss" to seconds.
    Example: 02:11:30 is 7890 seconds.

    :param timestamp: A timestamp of format "hh:mm:ss"
    :returns touple: A integer representing the timestamp in seconds.
    """
    h, m, s = map(int, timestamp.split(':'))
    return h * 3600 + m * 60 + s


def download_video(video_dir: str,
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
    if section_title in titles:
        titles[section_title] += 1
        suffix = titles[section_title]
        section_title = f"{section_title}-{suffix}"
    else:
        titles[section_title] = 0

    if not file_already_exists(f"{section_title}.mp4"):
        output = os.path.join(video_dir, section_title)

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
    # Get files from the download directory.
    # This allows adding new videos to the collection
    # without having to download them again.
    downloaded_files = get_all_files(video_download_path)
   
    try:
        with open(video_collection_path, "r") as file:
            video_collection: dict = yaml.safe_load(file)
    except Exception as e:
        print(f"Failed to load video collection YAML: {e}")
        return
    if video_download_path not in downloaded_files:
        os.mkdir(video_download_path) 
    for video in video_collection:
        video_title: str = video["title"]
        video_dir: str = os.path.join(video_download_path, video_title)
        if video_dir not in downloaded_files:
            os.mkdir(video_dir)
        video_url: str = video["url"]
        for section in video["sections"]:
            section_title: str = section["title"]
            start_time: int = timestamp_to_seconds(section["start"])
            end_time: int = timestamp_to_seconds(section["end"])
            download_video(video_dir, video_url, section_title, start_time, end_time)


def main():
    collect_videos()


if __name__ == "__main__":
    main()

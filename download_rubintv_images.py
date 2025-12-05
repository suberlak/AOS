import asyncio
import aiohttp
import os
import argparse
from tqdm import tqdm
import re

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Download PNG images with authorization and create a movie from them if requested.')
    parser.add_argument('dayobs', type=str, help='Observation date in YYYY-MM-DD or YYYYMMDD format.')
    parser.add_argument('image_type', type=str, help='Type of the image (e.g., "imexam").')
    parser.add_argument('--channel-name', type=str, default='lsstcam', help='Channel name (default: "lsstcam").')
    parser.add_argument('--maxseqnum', type=int, default=1000, help='Max sequence number for image count (default: 1000).')
    parser.add_argument('--video-name', type=str, help='Name of the output video file (if desired).')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the video (default: 30).')
    parser.add_argument('--download-dir', type=str, default='downloaded_pngs', help='Directory to save downloaded images.')
    return parser.parse_args()

def validate_dayobs(dayobs: str) -> str:
    """Validate and format the dayobs input."""
    if re.match(r'^\d{8}$', dayobs):
        return f"{dayobs[:4]}-{dayobs[4:6]}-{dayobs[6:]}"  # Convert to YYYY-MM-DD
    elif re.match(r'^\d{4}-\d{2}-\d{2}$', dayobs):
        return dayobs  # Already in YYYY-MM-DD
    else:
        raise ValueError("Date must be in YYYY-MM-DD or YYYYMMDD format.")

def get_base_url(dayobs: str, image_type: str, channel_name: str, ftype: str, use_rsp_dev: bool) -> str:
    """Construct the base URL for image downloading."""
    add = '-dev' if  use_rsp_dev else ''
    return f"https://usdf-rsp{add}.slac.stanford.edu/rubintv/event_image/summit-usdf/{channel_name}/{image_type}/{channel_name}_{image_type}_{dayobs}_{{:06d}}.{ftype}"

def create_download_directory(download_dir: str):
    """Create the directory for downloaded images."""
    os.makedirs(download_dir, exist_ok=True)

def read_auth_token(token_file: str) -> str:
    """Read the authorization token from a file."""
    with open(token_file, 'r') as f:
        return f.read().strip()  # Return the token without extra newlines

async def download_image(session: aiohttp.ClientSession, url: str, download_dir: str,filename: str, pbar: tqdm) -> bool:
    """Download an image and update the progress bar."""
    auth = aiohttp.BasicAuth(login=username, password=TOKEN)

    try:
        async with session.get(url, auth=auth) as response:
            if response.status == 200:
                file_path = os.path.join(download_dir, filename)
                with open(file_path, 'wb') as f:
                    f.write(await response.read())
                pbar.update(1)  # Successful download
                return True
            else:
                pbar.update(1)  # Failed download
                return False
    except Exception as e:
        pbar.update(1)  # Exception occurred
        return False

def create_movie(video_name: str, fps: int, download_dir: str,channel_name: str, image_type: str, dayobs: str, maxseqnum: int,  ftype: str):
    """Create a movie from the downloaded images."""
    import cv2  # Importing inside the function to avoid dependency issues

    images = []
    for i in range(maxseqnum):
        image_file = os.path.join(download_dir, f"{channel_name}_{image_type}_{dayobs}_{i:06d}.{ftype}")
        if os.path.exists(image_file):
            images.append(image_file)

    if len(images) == 0:
        print("No images to create a movie.")
        return

    # Read the first image to determine the width and height
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape

    # Create a VideoWriter object (for MP4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Progress bar for movie creation
    with tqdm(total=len(images), desc='Creating movie', unit='frame') as pbar:
        for image in images:
            frame = cv2.imread(image)
            video.write(frame)
            pbar.update(1)  # Update progress bar

    video.release()
    print(f"Movie created: {video_name}")

async def main(dayobs: str, image_type: str, channel_name: str, maxseqnum: int, video_name: str, fps: int, download_dir: str,  use_rsp_dev: bool):
    """Main function to download images."""
    # this builds the correct filename.
    # Need to check ahead of time - some images are sent to RubinTV as png, and some as jpg ... 
    # eg https://usdf-rsp-dev.slac.stanford.edu/rubintv/event_image/summit-usdf/ra_performance/aos_timing/ra_performance_aos_timing_2025-12-01_000375.jpg
    if channel_name == 'lsstcam_aos':
        ftype = 'png' 
    elif channel_name == 'lsstcam' or channel_name == 'ra_performance':
        ftype = 'jpg'
        
    base_url = get_base_url(dayobs, image_type, channel_name, ftype, use_rsp_dev)

    # Progress bar for downloading images
    with tqdm(total=maxseqnum, desc='Downloading images', unit='image') as pbar:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(maxseqnum):
                filename = f"{channel_name}_{image_type}_{dayobs}_{i:06d}.png"
                url = base_url.format(i)
                task = download_image(session, url, download_dir, filename, pbar)
                tasks.append(asyncio.create_task(task))

            await asyncio.gather(*tasks)

    if video_name:  # Only call create_movie if video_name is provided
        create_movie(video_name, fps, download_dir, channel_name, image_type, dayobs, maxseqnum, ftype)

# Read the username and password
username = "user"
token_file = os.path.expanduser("~/.lsst/rsp_token")
TOKEN = read_auth_token(token_file)

# Run the main function with command-line arguments
if __name__ == "__main__":
    args = parse_arguments()
    args.dayobs = validate_dayobs(args.dayobs)  # Validate and format dayobs
    create_download_directory(args.download_dir)

    asyncio.run(main(args.dayobs, args.image_type, args.channel_name, args.maxseqnum, args.video_name, args.fps, args.download_dir))


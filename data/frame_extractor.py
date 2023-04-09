import cv2
import numpy as np
import os
from pathlib import Path

from typing import List

FRAMES_PER_VIDEO = 100
VIDEO_SOURCE_FOLDER = "videos/"
IMAGE_DEST_FOLDER = "extracted_frames/"

def iterate_files(directory: str) -> List:
    """Iterates over the files in the given directory and returns a list of 
    found files."""
    files = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        fullpath = os.path.join(directory, filename)
        if os.path.isdir(fullpath):
            files += iterate_files(fullpath)
        else:
            files.append(fullpath)
    return files

class FrameExtractor:
        
    def __init__(self, video_source: str, verbose=False):
        self.source = video_source
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise Exception(f"{self.source} is not a valid video.")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.verbose = verbose
            
    def get_frames_evenly(self, n_frames: int) -> List:
        """Gets n_frames from the video evenly spaced throughout the video.
        
        If n_frames is greater than the number of frames in the video, all
        frames are extracted.
        """
        if n_frames <= 0:
            raise ValueError(f"'n_frames' must be 1 or more.")
        if n_frames < self.frame_count:
            step = int(self.frame_count/n_frames) # number of frames to skip
            frame_locations = np.arange(1, self.frame_count+1, step)
            frame_locations[:n_frames]
        else: # Get all frames from video
            frame_locations = np.arange(1,self.frame_count,1)
        return self.get_frames(frame_locations)
        
    def get_frames_per_second(self, fps: float) -> List:
        if fps == 0:
            if self.verbose:
                print("Cannot extract 0 frames per second.")
            return None
        frame_skip = round(round(self.fps)/fps)
        frame_locations = np.arange(0, self.frame_count, frame_skip)
        return self.get_frames(frame_locations)
        
    def get_frames(self, frame_locs: List[int]) -> List:
        """Returns a list of video frames as images or numpy arrays given the
        index position of the frame in the video."""
        # Reset vidcapture
        self.capture.set(1,0)
        frames = []
        n_success = 0
        for i in frame_locs:
            self.capture.set(1,i)
            success, image = self.capture.read()
            if success:
                n_success += 1
                frames.append({
                    'loc': i,
                    'frame': image
                })
        if self.verbose:
            print(f"Successfully extracted {n_success} frames from {self.source}")        
        return frames

def main():
    video_filenames = iterate_files(VIDEO_SOURCE_FOLDER)
    # Filter only mp4 files
    video_filenames = [f for f in video_filenames if f.endswith(".mp4")]
    
    count = 0
    for file_name in video_filenames:
        extractor = FrameExtractor(file_name)
        frames = extractor.get_frames_evenly(FRAMES_PER_VIDEO)
        
        vidname, _ = os.path.splitext(os.path.split(file_name)[-1])
        image_path_base = os.path.join(IMAGE_DEST_FOLDER, vidname)
        Path(image_path_base).mkdir(parents=True, exist_ok=True)
        
        for frame in frames:
            # Gets the frame_id and frame
            loc, img = frame['loc'], frame['frame']
            image_filename = f"{vidname}_{loc}.jpg"
            image_path = os.path.join(image_path_base, image_filename)
            cv2.imwrite(image_path, img)
            print(f"Saved {image_path}")
            count += 1
    
    print(f"Done extracting {count} frames from {len(video_filenames)} videos.")
    
    
if __name__ == "__main__":
    main()
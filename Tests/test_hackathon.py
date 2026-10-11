import sys
import os
from pathlib import Path
import time
import cv2

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.input_source import InputSource
from src.pipeline import process_frame, resize_for_display, DISPLAY_WIDTH
from src.postprocess import PipelineState

def test_videos(video_folder: str, max_videos=3, frames_per_vid=50):
    folder = Path(video_folder)
    videos = list(folder.glob("*.mp4"))
    if not videos:
        print(f"No videos found in {video_folder}")
        return
        
    print(f"Found {len(videos)} videos. Testing {min(max_videos, len(videos))} of them...")
    
    state = PipelineState(sensitivity=1.0, engine="model")
    
    results = []
    
    for i, vid_path in enumerate(videos[:max_videos]):
        print(f"\n--- Testing Video {i+1}: {vid_path.name} ---")
        try:
            source = InputSource(str(vid_path))
            state.set_source(source.source_desc, is_static_image=False)
            
            start_time = time.time()
            frames_processed = 0
            safe_frames = 0
            
            while frames_processed < frames_per_vid:
                ok, frame = source.read()
                if not ok or frame is None:
                    break
                    
                display_frame = resize_for_display(frame, DISPLAY_WIDTH)
                annotated = process_frame(display_frame, state)
                
                has_step = state.last_step is not None
                safe_to_step = has_step and state.last_step[2] < 0.6
                
                if safe_to_step:
                    safe_frames += 1
                    
                frames_processed += 1
                
            elapsed = time.time() - start_time
            fps = frames_processed / elapsed if elapsed > 0 else 0
            
            source.release()
            
            res = f"{vid_path.name}: {frames_processed} frames processed at {fps:.1f} FPS. Safe frames: {safe_frames}/{frames_processed}"
            print(res)
            results.append(res)
            
        except Exception as e:
            err = f"Error processing {vid_path.name}: {e}"
            print(err)
            results.append(err)
            
    print("\n=== SUMMARY OF RESULTS ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    test_videos(r"D:\IEEE Hackathon\Video")

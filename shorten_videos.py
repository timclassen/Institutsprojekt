import os
from yuv_io import read_yuv_video, write_yuv_video

def get_frames_from_yuv_video(video, start_idx, end_idx):
    return {"Y": video["Y"][start_idx:end_idx + 1], "U": video["U"][start_idx:end_idx + 1], "V": video["V"][start_idx:end_idx + 1]}


for filename in os.listdir("/home/staff/classen/Institutsprojekt/data/debug_videos"):
    f = os.path.join("/home/staff/classen/Institutsprojekt/data/debug_videos", filename)

    try:
        vid = read_yuv_video(f)
        vid_shortened = get_frames_from_yuv_video(vid, 0, 63)
        
        write_yuv_video(vid_shortened, "/home/staff/classen/Institutsprojekt/data/" + filename, automatic_file_name_extension=False)
    except Exception:
        pass
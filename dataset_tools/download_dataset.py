from tqdm import tqdm
import argparse
import os

def convert_video(video_name, dest_path, output_path):
    video_ext = video_name.split('.')[1]
    
    input_video = output_path + video_name
    output_video = dest_path + dest_path.split('/')[-2] + '.mp4'

    if video_ext == "mp4":
        cmd = "mv " + input_video + " " + output_video
    if video_ext == "mkv":
        cmd = "ffmpeg -loglevel error -i " + input_video + " -vcodec copy -acodec aac " + output_video
    if video_ext == "webm":
        cmd = "ffmpeg -loglevel error -i " + input_video + " -crf 10 -c:v libx264 " + output_video

    os.system(cmd)
    os.system("rm " + input_video)

def parse_args():
    parser = argparse.ArgumentParser(description='Download youtube videos from a txt file and organize files in l2d dataset style.')
    
    parser.add_argument("--input_txts", required=True, nargs='+',
                    help='path to txt file with url\'s of videos')
    parser.add_argument('--output_path', required=True,
                    help='Output path to create the dataset tree structure')

    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    output_path = args.output_path
    if output_path.rfind('/') != len(output_path)-1:
        output_path = output_path + '/'

    os.makedirs(args.output_path, exist_ok=True)

    n_styles = len(args.input_txts)
    styles = [style.split('/')[-1].split('.txt')[0] for style in args.input_txts]

    for idx, style in enumerate(styles):
        print("Downloading videos from style: " + style)

        style_videos = open(args.input_txts[idx], 'r').read().splitlines()

        ## Downloading video
        for i, video_url in enumerate(tqdm(style_videos, desc="Downloading videos...")):
            video_path = output_path + style + '_' + str(i)

            os.system("youtube-dl -q -i --no-warnings --format 'bestvideo+bestaudio' --add-metadata -o '" + video_path + ".%(ext)s' " + video_url)

        ## Listing directory style videos
        videos = [video for video in os.listdir(output_path) if len(video.split('.')) == 2]
        videos = [video for video in videos if video.split('_')[0] == style]

        ## Organizing dirs and converting videos to .mp4 format
        for i, video_name in enumerate(tqdm(videos, desc="Organizing style...")):
            dest_path = output_path + style + '/I_' + str(i) + '/'
            os.makedirs(dest_path, exist_ok=True)
            convert_video(video_name, dest_path, output_path)

if __name__ == "__main__":
    main()
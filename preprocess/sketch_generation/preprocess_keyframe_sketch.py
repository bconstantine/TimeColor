from informative_drawings.run import main as ID_main
import argparse
import os
import shutil

def main(args):
    ID_main(args)
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Sketch generation preprocessing.')
    parser.add_argument('--sketch_engine', help="Network to generate sketch", default='Anime2Sketch', 
                        choices=['Anime2Sketch', 'AniLines', 'informative_drawings'], type=str)
    parser.add_argument('--dataroot','-i', help='input folder or files', default='dataset/preprocess_done', type=str)
    parser.add_argument('--file_list', help='list of video files to process, should use input_type as video_text', default='dataset/preprocess_done', type=str)
    parser.add_argument('--output_folder', '-o', default='dataset/preprocess_sketch', type=str)
    parser.add_argument('--gpu_ids', '-g', default=['0'], help="gpu ids: e.g. 0 0,1,2 0,2.")
    parser.add_argument('--input_type', default="video", choices=['video', 'image', 'video_text'], type=str)
    #informative_drawings specific parameters
    parser.add_argument('--binarize_threshold', default=250, type=int, help="binarization threshold (out of 255, 0 to disable)")
    args = parser.parse_args()

    if type(args.gpu_ids) == str:
        args.gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    gpu_list = ','.join(str(x) for x in args.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    main(args)


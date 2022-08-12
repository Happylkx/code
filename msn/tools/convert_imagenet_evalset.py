import glob
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Convert official eval annotation to support mmselfsup')
    parser.add_argument('src_file', type=str,)
    parser.add_argument('data_path',type=str)
    parser.add_argument('out_file', type=str)

    args=parser.parse_args()
    return args

def main():
    args = parse_args()

    with open(args.src_file,'r') as f:
        src=f.readlines()

    # ['..../n03814639/ILSVRC2012_val_00018259.JPEG', '..../n03814639/ILSVRC2012_val_00019686.JPEG',...]
    print('Collecting '+os.path.join(args.data_path,'*/*.JPEG'))
    file_list=glob.glob(os.path.join(args.data_path,'*/*.JPEG'))
    map=dict()
    for file in file_list:
        tmp=file.split('/')
        dirname, filename=tmp[-2], tmp[-1]
        map[filename] = dirname+'/'+filename


    with open(args.out_file,'w') as f:
        for line in src:
            filename, cls = line.strip().split(' ')
            f.write(map[filename]+' '+cls+'\n')

if __name__ == '__main__':
    main()
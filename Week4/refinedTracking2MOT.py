import ast

def refined2MOT(path, output):

    frame_init = None

    with open(path, 'r') as f:
        lines = f.readlines()

        with open(output, 'w') as f:
            for line in lines:
                frame, camera, _, _, _, bb1, bb2, bb3, bb4, _, id= line.strip().split(',')

                if frame_init is None:
                    frame_init = frame

                bbox = bb1 + ',' + bb2 + ',' + bb3 + ',' + bb4

                bb = ast.literal_eval(bbox)

                f.write(f'{int(frame) - int(frame_init) + 1},{id},{bb[0]},{bb[1]},{bb[2]},{bb[3]},1,-1,-1,-1\n')

if __name__ == '__main__':
    input_path = 'seq3_tracking_50.txt'
    output_path = 'seq3_tracking_50_mot.txt'

    refined2MOT(input_path, output_path)

    '''
    python plot_bb_trail.py tracking_reid_mot.txt ../AIchallenge/train/S01/c001/vdo.avi out.mp4
    '''
'''
['91', 'c010', '1', '[-90.67348964765259', ' 42.49748343680444]', '[1283.42', ' 362.0', ' 235.93', ' 184.0]', 'S03Track\\c010\\frames\\frame00002.png', '1']
'''
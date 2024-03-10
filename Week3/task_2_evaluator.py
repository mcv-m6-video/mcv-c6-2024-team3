import subprocess
from pathlib import Path
import shutil
from tqdm import tqdm

from numpy import number

# https://pramod-atre.medium.com/understanding-object-tracking-a-hands-on-approach-part-1-3fb1afd0ae46

# Define the command you want to execute
command = 'python TrackEval/scripts/run_mot_challenge.py --BENCHMARK custom --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL PerfectTracker --METRICS HOTA Identity --USE_PARALLEL False --NUM_PARALLEL_CORES 1 --DO_PREPROC False'

GT_PATH = 'TrackEval/data/gt/mot_challenge/custom-train/custom-01/gt/gt.txt'
SEQINFO_PATH = 'TrackEval/data/gt/mot_challenge/custom-train/custom-01/seqinfo.ini'
PREDICTED_PATH = 'TrackEval/data/trackers/mot_challenge/custom-train/PerfectTracker/data/custom-01.txt'

def read_number_of_frames(path):
    with open(path, "r") as file:
        lines = file.readlines()

        last_line = lines[-1]

        last_line_values = last_line.strip().split(',')

        number_of_frames = int(float(last_line_values[0]))
    
    return number_of_frames


if __name__ == '__main__':

    folders = ['S01Track', 'S03Track', 'S04Track']
    # folders = ['S04Track']

    for folder in folders:
        folderPath = Path(folder)

        for sequence in tqdm(folderPath.iterdir()):
            gt = sequence / 'gt.txt'
            deep_sort = sequence / 'tracking_DeepSort.txt'
            normal_sort = sequence / 'tracking_SORT.txt'

            gt_frames = read_number_of_frames(gt)
            deep_sort_frames = read_number_of_frames(deep_sort)
            normal_sort_frames = read_number_of_frames(normal_sort)

            number_of_frames = max(gt_frames, deep_sort_frames, normal_sort_frames)

            with open(SEQINFO_PATH, "w") as file:
                file.write("[Sequence]\n")
                file.write("name=custom\n")
                file.write("seqLength= {}\n".format(number_of_frames))
            
            shutil.copy(gt, GT_PATH)
            shutil.copy(deep_sort, PREDICTED_PATH)
            completed_process_deep = subprocess.run(command, shell=True, capture_output=True, text=True)

            shutil.copy(normal_sort, PREDICTED_PATH)
            completed_process_sort = subprocess.run(command, shell=True, capture_output=True, text=True)

            with open(sequence / 'deepsort_eval.txt', 'w') as file:
                file.write(completed_process_deep.stdout)
                file.write(completed_process_deep.stderr)
            
            with open(sequence / 'sort_eval.txt', 'w') as file:
                file.write(completed_process_sort.stdout)
                file.write(completed_process_sort.stderr)
            






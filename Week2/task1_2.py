import os

from utils import reset_folder, save_frames

if __name__ == "__main__":
    # S01, S03, S04
    # S01: c001, c002, c003, c004, c005
    # S03: c010, c011, c012, c013, c014, c015
    # S04: c016, c017, c018, c019, c020, c021, c022, c023, c024, c025, c026, c027, c028, c029, c030, c031, c032, c033, c034, c035, c036, c037, c038, c039, c040

    input_video = '../OptionalTaskW2Dataset/train/S01/c001/vdo.avi'
    output_frames = 'S01_c001'

    # Estoy suponiendo que ya he generado las imagenes pa no tener que hacerlo otra vez
    save_frames(input_video, output_frames)
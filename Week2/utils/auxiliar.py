from asyncore import loop
import imageio
from PIL import Image
import numpy as np

in_files = ['finetuned.gif', 'pretrained.gif']
out_files = ['finetuned_short.gif', 'pretrained_short.gif']

for input_gif_path, output_gif_path in zip(in_files, out_files):
    # Open the input GIF
    input_gif = imageio.get_reader(input_gif_path)

    # Calculate the new duration for each frame
    new_duration = input_gif.get_meta_data()['duration'] // 2  # Faster playback

    # Create a writer object to save the resized GIF
    writer = imageio.get_writer(output_gif_path, duration=new_duration, loop=0)

    # Resize each frame to a quarter of its original size and write to the output GIF
    for frame in input_gif:
        resized_frame = Image.fromarray(frame)

        resized_frame = resized_frame.resize((resized_frame.size[0] // 3, resized_frame.size[1] // 3))
    
        # Convert the resized frame to numpy array
        resized_frame = np.array(resized_frame)

        writer.append_data(resized_frame[:, :, ::-1])

    # Close the writer and the original GIF
    writer.close()
    input_gif.close()

import imageio

# Open the GIF
gif_path = "sigrid2.gif"
gif = imageio.get_reader(gif_path)

# Calculate the new duration for each frame
new_duration = gif.get_meta_data()['duration'] * 2

# Create a writer object to save the resized GIF
output_path = "sigrid2_short.gif"
writer = imageio.get_writer(output_path, duration=new_duration)

# Iterate over each frame, but use every other frame
for i, frame in enumerate(gif):
    if i % 6 == 0:
        writer.append_data(frame)

# Close the writer and the original GIF
writer.close()
gif.close()
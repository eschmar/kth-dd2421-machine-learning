from os import listdir
from os.path import isfile, join
import imageio

frames = [f for f in listdir("out/") if isfile(join("out/", f)) and f.startswith("frame")]

images = []
for frame in frames:
    images.append(imageio.imread(join("out/", frame)))

imageio.mimsave('out/movie.gif', images)
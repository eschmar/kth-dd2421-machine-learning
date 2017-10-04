from os import listdir
from os.path import isfile, join
import imageio
import sys, helper, main

# Run
if __name__ == '__main__':
    params = helper.parseArguments(sys.argv)

    if '--gif' not in params:
        frames = []
        for i in (x / 10 for x in range(1, 100)):
            current = "frame{0}.png".format('%03d' % int(i * 10))
            main.run(['--radial', '-sigma', str(i), '--save', '-o', current])
            frames.append(current)
    else:
        frames = [f for f in listdir("out/") if isfile(join("out/", f)) and f.startswith("frame")]

    images = []
    for frame in frames:
        images.append(imageio.imread(join("out/", frame)))

    imageio.mimsave('out/movie.gif', images)

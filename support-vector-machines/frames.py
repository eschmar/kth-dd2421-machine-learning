import sys, helper, main, subprocess

# Run
if __name__ == '__main__':
    params = helper.parseArguments(sys.argv)

    for i in (x / 10 for x in range(1, 100)):
        current = '%03d' % int(i * 10)
        main.run(['--radial', '-sigma', str(i), '--save', '-o', "frame{0}.png".format(current)])

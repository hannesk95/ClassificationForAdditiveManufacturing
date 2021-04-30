import subprocess


#cat stl2_v00.txt | xargs -n 2 -P 8 sh -c 'wget --no-check-certificate $0 -O stl/$1'
#cat stl2_v00.txt | xargs -n 2 -P 8 sh -c 'wget --no-check-certificate $0 -O stl2/$1'

command = "cat" + " stl2_v00.txt" + " |" + " xargs" + " -n" + " 2" + " -P" + " 8" + " sh" " -c" + " 'wget --no-check-certificate $0 -O stl2/$1'"

process = subprocess.Popen(command)



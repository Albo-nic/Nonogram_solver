from timeit import timeit
import main
import matplotlib.pyplot as plt
import numpy as np


#################### Parameters ####################

do_hybrid = True # Test "hybrid_solve"?
do_cell = False # Test "cell_solve"?
do_row = False # Test "row_solve"?

size = 10 # Side length of tested nonograms
repeat = 50 # Number of times each fill is tested

start = 0 # Lowest fill value to be tested
stop = 1 # Highest fill value to be tested
number = 4 # Number of tested fill values between "start" and "stop", int greater than 1

####################################################


h_ypoints = []
c_ypoints = []
r_ypoints = []
x_points = []


for x in np.linspace(start, stop, num=number):
    if do_hybrid == True:
        h_ypoints.append(timeit(f"a = main.random_nonogram({size}, {size}, {x})\nmain.hybrid_solve(a[0], a[1])", number=repeat, setup="import main")/repeat)
    if do_cell == True:
        c_ypoints.append(timeit(f"a = main.random_nonogram({size}, {size}, {x})\nmain.recursive_cell_solve(a[0], a[1])", number=repeat, setup="import main")/repeat)
    if do_row == True:
        r_ypoints.append(timeit(f"a = main.random_nonogram({size}, {size}, {x})\nmain.recursive_row_solve(a[0], a[1])", number=repeat, setup="import main")/repeat)
    x_points.append(x)
    print(x)


if do_hybrid == True:
    plt.plot(x_points, h_ypoints, label="hybrid_solve")
if do_cell == True:
    plt.plot(x_points, c_ypoints, label="recursive_cell_solve")
if do_row == True:
    plt.plot(x_points, r_ypoints, label="recursive_row_solve")


plt.xlabel("Puzzle fill")
plt.ylabel("Average time duration of solution")
plt.legend()

plt.show()
import timeit
import main
import matplotlib.pyplot as plt
print("Program started")


#################### Parameters ####################

repeat = 10 # Number of times each size is tested
fill = 0.5 # Fill of tested nonograms

max_hybrid_size = 20 # maximum size of nonogram soleved with "hybrid_solve"
max_row_size = 7 # maximum size of nonogram soleved with "recursive_row_solve"
max_cell_size = 6 # maximum size of nonogram soleved with "recursive_cell_solve"

####################################################


xpoints = [x for x in range(1,max_hybrid_size+1)]
ypoints = [timeit.timeit(f"p = main.random_nonogram({x},{x},{fill})\nmain.hybrid_solve(p[0], p[1])", setup=f"import main\nprint({x-1})", number=repeat)/repeat for x in range(1,max_hybrid_size+1)]
plt.plot(xpoints, ypoints, label="hybrid_solve")
print("hybrid_solve done!")

xpoints = [x for x in range(1,max_row_size+1)]
ypoints = [timeit.timeit(f"p = main.random_nonogram({x},{x},{fill})\nmain.recursive_row_solve(p[0], p[1])", setup=f"import main\nprint({x-1})", number=repeat)/repeat for x in range(1,max_row_size+1)]
plt.plot(xpoints, ypoints, label="recursive_row_solve")
print("recursive_row_solve done!")

xpoints = [x for x in range(1,max_cell_size+1)]
ypoints = [timeit.timeit(f"p = main.random_nonogram({x},{x},{fill})\nmain.recursive_cell_solve(p[0], p[1])", setup=f"import main\nprint({x-1})", number=repeat)/repeat for x in range(1,max_cell_size+1)]
plt.plot(xpoints, ypoints, label = "recursive_cell_solve")
print("recursive_cell_solve done!")

plt.title("Time duration of puzzle solution with regard to side length of puzzle")
plt.xlabel = "Side length"
plt.ylabel = "Average time duration of solution"

plt.legend()
plt.show()

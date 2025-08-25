This project is a solver for nonogram puzzles (more information on https://en.wikipedia.org/wiki/Nonogram) in python. The program can tell if no solution exists and if there are several solutions, it will find only one of them. The program can also plot the solution and several time complexity tests are included

**Overview of some functions in main.py**

**hybrid_solve(r_blocks: tuple[tuple[int]], c_blocks: tuple[tuple[int]]):**
Accepts two tuples as input as instruction. For example, the nonogram

<img width="228" height="216" alt="image" src="https://github.com/user-attachments/assets/a0574108-3fa8-4226-8077-11c6408379d1" />

would be inputted as "hybrid_solve(((1,), (2,), (4,), (1, 1), (1, 1)), ((1,), (5, ), (1,), (3,), (1,)))".
This function returns a 2-dimensional np.array with 1 for full cells and -1 for empty cells. In our case, we get:

array([[-1,  1, -1, -1, -1],
       [ 1,  1, -1, -1, -1],
       [-1,  1,  1,  1,  1],
       [-1,  1, -1,  1, -1],
       [-1,  1, -1,  1, -1]], dtype = int8)

 If no solution exists, the function returns None

 How it works: Firstly, we generate all possible ways each row/column could be filled based on the given instructions for that row column. After that, the program alternates between two phases:
 1. row_solve: The program repeatedly cycles through all rows and columns. For each, it finds all solutions to that row/column that are no longer possible and removes them from that row's list of possible solutions. After that it finds all cells in that row/column that are either empty or filled in all possible solutions, and declares them filled/empty. Once it completes the full cycle without changing any cells, it returns a (partially) completed puzzle, or it finds that no solution exists
 2. DFS: If given a partial solution, the program guesses a cell, adds this state and which cell it guessed to a stack and continues. If row_solve finds no solution exists, it reverts to last saved state and changes its guess. If no such state exists, the puzzle is unsolvable.

On complexity: Nonograms are an NP-complete problem, and the time complexity of this algorithm is exponential. If you find a solution with polynomial time complexity, please let me know.

**recursive_cell_solve(r_blocks: tuple[tuple[int]], c_blocks: tuple[tuple[int]]), recursive_row_solve(r_blocks: tuple[tuple[int]], c_blocks: tuple[tuple[int]]):**
These functions solve the nonogram either by guessing cells in DFS (recursive_cell_solve) or generating all possible row solutions and guessing from those 
(recursive_row_solve). Both are significantly slower than hybrid_solve when solving bigger puzzles.

**plot_solution(matrix):**: If given None, does nothing, if given a 2-dimensional np.array containing 1 and -1, it plots the solution. In the previous example "plot_solution(hybrid_solve(((1,), (2,), (4,), (1, 1), (1, 1)), ((1,), (5, ), (1,), (3,), (1,))))", shows:

<img width="220" height="217.2" alt="image" src="https://github.com/user-attachments/assets/79103424-c645-4800-918e-b318adfa28ed" />

**random_nonogram(width, height, fill=0.5):** Returns a randomly generated puzzle with given width and height and the proportion of filled cells of "fill". The function returns a tuple containing the instructions for a rows, then the columns. It does this by generating a matrix with a given amount of filled cells.

**check_solution(matrix, r_blocks, c_blocks):** Given a matrix and instructions, this function checks if matrix is soulution to given instruction. If this is the case, returns True, otherwise returns False. In our example,

check_solution(np.array([[-1,  1, -1, -1, -1],
       [ 1,  1, -1, -1, -1],
       [-1,  1,  1,  1,  1],
       [-1,  1, -1,  1, -1],
       [-1,  1, -1,  1, -1]]), ((1,), (2,), (4,), (1, 1), (1, 1)), ((1,), (5, ), (1,), (3,), (1,)))
       
returns True

**fill_test.py and size_test.py**

Also included are two programs for comparing time duration of different solution methods with regard to proportion of filled cells or size of solved puzzle. Various parameters that can be tweaked are explained in the comments of the code.

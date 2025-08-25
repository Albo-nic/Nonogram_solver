import matplotlib.patches
import numpy as np
from doctest import testmod
import matplotlib.pyplot as plt
from copy import deepcopy
from copy import copy
from random import sample

#counter = 0
#incorrect_guesses = 0

def initial_checks(c_blocks: tuple[tuple[int]], r_blocks: tuple[tuple[int]]) -> bool:
    """A few initial checks that the puzzle is solvable. Returns False if found unsolvable, true otherwise"""

    if sum([sum(b) for b in c_blocks]) != sum([sum(b) for b in r_blocks]): # Row and column instructions need same amount of filled cells
        return False
    
    width = len(c_blocks)
    height = len(r_blocks)

    for item in r_blocks: # Row instructions fit into the puzzle
        if sum(item)+len(item)-1 > width:
            return False
        
    for item in c_blocks: # Column instructions fit into the puzzle
        if sum(item) + len(item) -1 > height:
            return False
        
    return True


def row_repr(blocks: tuple[int], indices: list[int], length: int) -> tuple[int]:
    """
    Given list of blocks, starting indices and length returns literal representation of row
    >>> row_repr((1,3), [1,4], 8)
    (-1, 1, -1, -1, 1, 1, 1, -1)
    """
    if len(blocks) != len(indices):
        raise ValueError("Blocks and Indices must have same length")
    
    row = [-1 for x in range(length)]
    pointer = 0
    for i in range(len(indices)):
        pointer = indices[i]
        for cell in range(blocks[i]):
            row[pointer] = 1
            pointer += 1
    
    return tuple(row)


def all_solutions(blocks: tuple[int], length: int) -> set[tuple[int]]:
    """
    Returns set of all possible solutions of blocks on line of given length
    >>> s = all_solutions((1,2), 5)
    >>> s == {(1, -1, 1, 1, -1), (1, -1, -1, 1, 1), (-1, 1, -1, 1, 1)}
    True
    >>> all_solutions((3,1), 4)
    set()
    """



    def nudges2(length: int, max_n: int, beginnings=[]) -> set[tuple[int]]:
        """
        Returns set of non-decreasing tuples of non-negative integers with maximum value max_n
        >>> nudges2(2, 2)
        {(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)}
        """

        if beginnings == []:
            for x in range(max_n+1):
                beginnings.append([x])
            return nudges2(length-1, max_n, beginnings)

        elif length == 0:
            return beginnings

        else:
            new = []
            for item in beginnings:
                for next_step in range(item[-1], max_n + 1):
                    new.append(item + [next_step])
            return nudges2(length-1, max_n, new)
        


    def elementwise_add(l: list[int], m:list[int]) -> list[int]:
        """Adds two lists together by elements"""

        if len(l) != len(m): raise ValueError("Lists of different length cannot be added together")
        return [l[i]+m[i] for i in range(len(l))]

    if blocks == (): # If given an empty row
        return{tuple([-1 for i in range(length)])}

    starting_indices = [0]

    for item in blocks:
        starting_indices.append(1 + item + starting_indices[-1])
    starting_indices.pop()

    wiggle_room = length - starting_indices[-1] - blocks[-1]
    if wiggle_room < 0:
        return set()

    possible_placements = nudges2(len(blocks), wiggle_room)
    for i in range(len(possible_placements)):
        possible_placements[i] = elementwise_add(possible_placements[i], starting_indices)

    # Currently gives starting indices

    for i in range(len(possible_placements)):
        possible_placements[i] = row_repr(blocks, possible_placements[i], length)

    return set(possible_placements)



def intersect_rows(rows: set[tuple[int]]) -> tuple[int]:
    """
    Returns tuple with: 1 (-1) wherever all lists in tuples in rows have 1 (-1), and 0 everywhere else
    >>> intersect_rows(frozenset({(1, 1, -1), (1, -1, -1)}))
    (1, 0, -1)
    >>> intersect_rows(frozenset({(1, 1), (1, -1), (-1, -1)}))
    (0, 0)
    """

    if len(rows) == 0:
        raise ValueError("No rows given")
    
    common = []
    for item in rows:
        if common == []:
            common = list(item)
            non_zero = len(common) # Number of non-zero items in common
        else:
            for j in range(len(common)):
                if common[j] == 0:
                    pass
                elif common[j] in [-1, 1]:
                    if common[j] != item[j]:
                        common[j] = 0
                        non_zero += -1
                else:
                    # common[j] = item[j]
                    pass
                if non_zero == 0: # end program early
                    return tuple(common) 
    return tuple(common)



class solve_state():
    def __init__(self, r_blocks: tuple[tuple[int]], c_blocks: tuple[tuple[int]]):

        self.r_blocks = r_blocks
        self.height = len(r_blocks)

        self.c_blocks = c_blocks
        self.width = len(c_blocks)

        self.possible_rows = tuple([all_solutions(r_blocks[i], self.width) for i in range(len(r_blocks))])
        #self.row_intersections = intersect_rows(self.possible_rows)
        self.rows_changed_since_last = [True for item in range(self.height)]

        self.possible_columns = tuple([all_solutions(c_blocks[i], self.height) for i in range(len(c_blocks))])
        #self.column_intersections = intersect_rows(self.possible_columns)
        self.columns_changed_since_last = [True for item in range(self.width)]

        self.zeros_in_row = [self.width for row in r_blocks] # number of zeros in row
        self.zeros_in_column = [self.height for column in c_blocks]

        self.matrix = np.zeros((self.height, self.width), dtype=np.int8)
    
    def __repr__(self):
        return str(self.matrix)



def block_repr(l: tuple) -> tuple:
    """
    Returns block representation of a solved row/column
    >>> block_repr((1, -1))
    (1,)
    >>> block_repr((1, 1, -1, 1, -1))
    (2, 1)
    """
    b=[]
    in_block = False
    for cell in l:
        if cell == -1:
            in_block = False
        elif cell == 1:
            if in_block == False:
                b.append(1)
                in_block = True
            else:
                b[-1] += 1
        else:
            raise ValueError("block_repr: input may only contain 1 or -1")
    b = tuple(b)
    return b



def find_zero(matrix: np.array):
    """Finds indices of 0 in matrix, if none exists return None"""

    for x in range(len(matrix[0])):
        for y in range(len(matrix)):

            if matrix[y][x] == 0:
                return (y,x)
    
    return None



def row_solve(state: solve_state) -> solve_state:
    """
    Solve puzzle using row method while possible. Returns partial solution in form (matrix, possible_rows, possible_columns) or None is unsolvable
    """


    def conflicting(r1, r2) -> bool:
        """
        Recieves two rows as input, if an index exists such that r1[i] == 1 and r2[i] == -1 or vice versa, returns False, otherwise returns True
        >>> conflicting([1, 0, -1], [1, -1, 0])
        False
        >>> conflicting([0, 1, 0], [0, -1, 1])
        True
        """
        if len(r1) != len(r2):
            raise ValueError("Different lengths of rows in conflicting")
        
        for i in range(len(r1)):
            if r1[i] in [-1, 1] and r2[i] in [-1, 1] and r1[i] != r2[i]:
                return True
        return False
    

    while True:
        anything_changed = False

        # Go through columns
        for c_index in range(state.width):

            # Each column
            to_remove = []
            # Find which columns need to be reconsidered
            if state.columns_changed_since_last[c_index] == True:
                
                # Find and remove conflicting possibilities
                for possibility in state.possible_columns[c_index]:
                    if conflicting(state.matrix[:,c_index], possibility) == True:
                        to_remove.append(possibility)
                for impossibility in to_remove:
                    state.possible_columns[c_index].remove(impossibility)

                # No possible solutions
                if len(state.possible_columns[c_index]) == 0:
                    return None
                
                intersection = intersect_rows(state.possible_columns[c_index])

                for i in range(len(intersection)): # Go through items in intersection, see if progress has been made
                    if intersection[i] in [-1, 1] and state.matrix[:,c_index][i] == 0: # Progress made
                        anything_changed = True
                        state.rows_changed_since_last[i] = True # Row needs to be recalculated
                        state.matrix[:,c_index][i] = intersection[i]

        state.columns_changed_since_last = [False for item in state.possible_columns]

        # Go through rows
        for r_index in range(state.height):

            # Each column
            to_remove = []
            if state.rows_changed_since_last[r_index] == True: # Something has changed in row and options need to be reconsidered
            
                for possibility in state.possible_rows[r_index]: # Go through all possibilities and remove the ones that aren't viable
                    if conflicting(state.matrix[r_index], possibility) == True:
                        to_remove.append(possibility)
                for impossibility in to_remove:
                    state.possible_rows[r_index].remove(impossibility)

                if len(state.possible_rows[r_index]) == 0: # No possible solutions to row
                    return None
                
                intersection = intersect_rows(state.possible_rows[r_index])

                for i in range(len(intersection)): # Go through items in intersection, see if progress has been made
                    if intersection[i] in [-1, 1] and state.matrix[r_index][i] == 0: # Progress made
                        anything_changed = True
                        state.columns_changed_since_last[i] = True # Column needs to be recalculated
                        state.matrix[r_index][i] = intersection[i]

        state.rows_changed_since_last = [False for item in state.possible_rows]

        if anything_changed == False:
            break
        
    return (state)



def hybrid_solve(r_blocks: tuple[tuple[int]], c_blocks: tuple[tuple[int]]):
    """
    Solve puzzle using row_solve where possible and DFS where not. Returns None if solution doesn't exist
    """
    
    if initial_checks (c_blocks, r_blocks) == False:
        return None
    
    #global counter
    #global incorrect_guesses

    state = solve_state(r_blocks, c_blocks)
    #print("Generated all solutions for rows/columns")

    guessed_cells = []
    saved_states = []

    while True:
        state = row_solve(state)

        if state == None:
            if guessed_cells == []: # No guesses have been made, therefore no mistakes could have been made
                return None
            # Last guess was incorrect, revert to last functioning state and declare guessed as -1
            #incorrect_guesses += 1 # For testing
            state = saved_states.pop()
            incorrect_guess = guessed_cells.pop()
            state.matrix[incorrect_guess[0]][incorrect_guess[1]] = -1

            state.rows_changed_since_last[incorrect_guess[0]] = True
            state.columns_changed_since_last[incorrect_guess[1]] = True
        
        else: # Guess a cell is 1
            #print(state.matrix)
            #counter += 1
            #print(counter)
            saved_states.append(deepcopy(state))
            # Find an unknown cell, put 1 in and see what happens. If none such cell exists we have found a solution (yay)
            indices = find_zero(state.matrix)
            
            if indices == None:
                return state.matrix
            
            guess = (indices[0], indices[1])
            guessed_cells.append(deepcopy(guess))
            state.matrix[guess[0]][guess[1]] = 1

            # The row and column of guessed cell need to be revised
            state.rows_changed_since_last[guess[0]] = True
            state.columns_changed_since_last[guess[1]] = True



def recursive_cell_solve(r_blocks: tuple[tuple[int]], c_blocks: tuple[tuple[int]]):
    """Solves nonogram using only DFS by guessing cells"""

    if initial_checks (c_blocks, r_blocks) == False:
        return None
    
    width = len(c_blocks)
    height = len(r_blocks)
    
    stack = [np.zeros((height, width), dtype=np.int8)]
    zero_coords = [[0,0]] # These coordinates correspond to the location of a zero for each matrix in stack
    
    while True:

        if len(stack) == 0: # No solution exists
            return None

        current = stack.pop()
        zero_location = zero_coords.pop()

        filled_branch = current.copy()
        filled_branch[zero_location[0]][zero_location[1]] = 1

        empty_branch = current.copy()
        empty_branch[zero_location[0]][zero_location[1]] = -1

        if zero_location[1] != width - 1: # Not in last column:
            stack.append(filled_branch.copy())
            stack.append(empty_branch.copy())

            zero_location[1] += 1 # Move one to the right
            for x in range(2):
                zero_coords.append(zero_location.copy())


        elif zero_location[0] != height - 1: # In last column, not in last row (row has been completed)

            if block_repr(filled_branch[zero_location[0]]) == r_blocks[zero_location[0]]:
                stack.append(filled_branch)

                zero_location[0] += 1
                zero_location[1] = 0
                zero_coords.append(zero_location.copy())

            elif block_repr(empty_branch[zero_location[0]]) == r_blocks[zero_location[0]]: # They can't both be right
                stack.append(empty_branch)

                zero_location[0] += 1
                zero_location[1] = 0
                zero_coords.append(zero_location.copy())

        else: # Complete solution

            for solution in [filled_branch, empty_branch]:

                if block_repr(solution[-1]) == r_blocks[-1]: # Last row is correct
                    is_solution = True
                    for c_index in range(width):
                        if block_repr(solution[:,c_index]) != c_blocks[c_index]:
                            is_solution = False
                            break
                    
                    if is_solution:
                        return solution
                    


def recursive_row_solve(r_blocks: tuple[tuple[int]], c_blocks: tuple[tuple[int]]):
    """Solves nonogram by using DFS by guessing entire rows"""

    if initial_checks (c_blocks, r_blocks) == False:
        return None
    
    width = len(c_blocks)
    height = len(r_blocks)
    
    stack = [np.zeros((height, width), dtype=np.int8)]
    zero_rows = [0] # These coordinates correspond to the location the first empty row for each matrix in stack

    possible_rows = [all_solutions(r_blocks[i], width) for i in range(height)]

    while True:
        if len(stack) == 0:
            return None
        
        matrix = stack.pop()
        row_index = zero_rows.pop()
        #print(matrix)

        if row_index == height-1: # Final row
            for item in possible_rows[row_index]:
                matrix[row_index] = item # Generates all possible solutions

                is_solution = True # Checks validity of solution
                for c_index in range(width):
                    if block_repr(matrix[:,c_index]) != c_blocks[c_index]:
                        is_solution = False
                        break
                
                if is_solution:
                    return matrix
        
            for item in possible_rows[row_index]:
                matrix[row_index] = item # Generates all possible solutions

        else: # Not the final row
            for item in possible_rows[row_index]:
                matrix[row_index] = item # Generates all possible solutions
                stack.append(matrix.copy())
                zero_rows.append(row_index+1)



def plot_solution(matrix) -> None:
    """Plots given matrix on a grid"""
    if type(matrix) == type(None):
        return
    a = max(len(matrix), len(matrix[0]))
    plt.figure(figsize = (6*len(matrix[0])/a, 6*len(matrix)/a)) # The longer side of the diagram will always be 6 in

    x_values = []
    for c_index in range(len(matrix[0])):
        x_values.append(block_repr(matrix[:,c_index]))
    
    y_values = []
    for r_index in range(len(matrix)-1, -1, -1):
        y_values.append(block_repr(matrix[r_index]))


    plt.xticks(list(range(1, len(matrix[0])+1)), x_values)
    plt.yticks(list(range(1, len(matrix)+1)), y_values)

    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.set_xlim(0.5, len(matrix[0]) + 0.5)
    ax.set_ylim(0.5, len(matrix) + 0.5)
    for x in range(len(matrix[0])):
        for y in range(len(matrix)):
            if matrix[y][x] == 1:
                s = matplotlib.patches.Rectangle((x+0.5, len(matrix)-y-0.5),1,1)
                ax.add_patch(s)
    plt.show()
    return


def random_nonogram(width, height, fill=0.5) -> tuple[tuple[tuple[int]]]:
    """Generates a nonogram puzzle with given width and height where fill of all cells are filled in, for testing. Returns (row_blocks, column_blocks)"""

    cell_list = [n for n in range(width * height)]
    to_fill = sample(cell_list, round(width*height*fill))

    matrix = np.full((height, width), -1, dtype=np.int8)

    for item in to_fill:
        matrix[item // width][item % width] = 1

    c_blocks = []
    for c_index in range(width):
        c_blocks.append(block_repr(matrix[:,c_index]))

    r_blocks = []
    for r_index in range(height):
        r_blocks.append(block_repr(matrix[r_index]))

    return (tuple(r_blocks), tuple(c_blocks))



def check_solution(matrix, r_blocks, c_blocks):
    """Checks if matrix is a solution to puzzle"""
    for r_index in range(len(matrix)):
        if r_blocks[r_index] != block_repr(matrix[r_index]):
            return False
    
    for c_index in range(len(matrix[0])):
        if c_blocks[c_index] != block_repr(matrix[:,c_index]):
            return False
        
    return True


#################### Example program ####################

if __name__ == "__main__":

    a = random_nonogram(15, 15, 0.3)
    h = hybrid_solve(a[0], a[1])
    plot_solution(h)
# Solving Sudoku Puzzle
This project is able to solve easy-to-medium level Sudoku puzzles where there is no need to choose between two or more possibilities in each puzzle grid.

## How to play
Sudoku is played on a 9x9 grid and it is also separated into 9 boxes (each box is a 3x3 grid). The initial configuration of Sudoku is a partially filled board. We need to figure out how to put the digits 1-9 on the grid so that each digit appears exactly once in each row, column, and box.

## Methods to solve the puzzle
I used a CNN model to recognize the initial handful numbers on the board, then used contours to find the coordination of them.  
The algorithm used to solve the puzzle: I started from the topleft box, searching for a grid that have only one possibility, then moved to the next box and did the same, till the last box. I repearted this action a few times to ensure all the grids would fill.

## How to use this repo
Input: a sudoku puzzle image.  
output: a completed sudoku puzzle image.

![git](https://user-images.githubusercontent.com/103570811/179358848-96765be3-0b94-4f75-8e8c-cc6a8a29eb0c.png)

You are required to enter two quantities:  
1. Sudoku image address.  
2. Thresh_amount.  
If the image has black lines, the tresh_amount variable should take the amount of 200. And if the image has light gray lines, the tresh_amount variable should take the amount of 240 or 250.  
  
  Problem: when the CNN model is not able to recognize all initial numbers on the board correctly, the rest of the program will not be able to solve the puzzel completely and correctly.

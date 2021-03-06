# Solving Sudoku Puzzle
This project is able to solve easy-to-medium level Sudoku puzzles where there is no need to choose between two or more possibilities in each puzzle grid.

## How to play
Sudoku is played on a 9x9 grid and it is also separated into 9 boxes (each box is a 3x3 grid). The initial configuration of Sudoku is a partially filled board. We need to figure out how to put the digits 1-9 on the grid so that each digit appears exactly once in each row, column, and box.

## Methods to solve the puzzle
I used a CNN model to recognize the initial handful numbers on the board, and contours to find the coordination of them.  
The algorithm used to solve the puzzle: I started from the topleft box, searching for a grid that has only one possibility, then moved to the next box and did the same, till the last box. I repeated this action a few times to ensure all the grids would fill.

## How to use this repo
The main.py contains the code to solve the puzzle. The cnn9k.h5 is the saved model used in the code to predict the initial numbers on the board, and simple_sudoku.png is an example picture of sudoku puzzle using for test of the model.  

  Input: a sudoku puzzle image (the image should not have title or written texts on or around the puzzle).  
Output: a completed sudoku puzzle image.

![git](https://user-images.githubusercontent.com/103570811/179358848-96765be3-0b94-4f75-8e8c-cc6a8a29eb0c.png)

You are required to enter two quantities:  
1. Sudoku image address.  
2. Thresh_amount.  
If the image has black lines, the tresh_amount variable should take the amount of 200. And if the image has light gray lines, the tresh_amount variable should take the amount of 240 or 250. You better try numbers from 200 to 250 to see which will work because thresh_amount plays an important role to predict numbers correctly.
  
  Problem: in some cases, when the CNN model is not able to recognize all initial numbers on the board correctly, the rest of the program will not be able to solve the puzzel completely and correctly.

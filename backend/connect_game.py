def create_board():
    """Create a 7x6 board for Connect 4."""
    return [[0 for _ in range(7)] for _ in range(6)]

def make_move(board, column, player):
    """Attempts to make a move for the player in the given column."""
    for row in reversed(range(6)):
        if board[row][column] == 0:
            board[row][column] = player
            return True
    return False  # Column is full

def check_win(board):
    """Checks the board for a winning condition."""
    # Check horizontal, vertical, and two diagonal directions
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for row in range(6):
        for col in range(7):
            if board[row][col] == 0:
                continue
            for d in directions:
                dr, dc = d
                if all(0 <= row + i*dr < 6 and 0 <= col + i*dc < 7 and board[row + i*dr][col + i*dc] == board[row][col] for i in range(4)):
                    return board[row][col]
    return 0  # No winner

def check_full(board):
    """Checks if the board is full."""
    return all(board[0][col] != 0 for col in range(7))

def display_board(board):
    """Displays the current state of the board."""
    print("\n  0 1 2 3 4 5 6")
    print("  ------------")
    for row in board:
        print('|' + ' '.join('O' if cell == 1 else 'X' if cell == 2 else ' ' for cell in row) + '|')
    print("  ------------")

def play_game():
    """The main game loop."""
    board = create_board()
    current_player = 1
    while True:
        display_board(board)
        try:
            column = int(input(f"Player {'O' if current_player == 1 else 'X'}, choose column (0-6): "))
        except ValueError:
            print("Please enter a valid integer between 0 and 6.")
            continue
        if 0 <= column <= 6:
            if make_move(board, column, current_player):
                if check_win(board):
                    display_board(board)
                    print(f"Player {'O' if current_player == 1 else 'X'} wins!")
                    break
                if check_full(board):
                    display_board(board)
                    print("It's a tie!")
                    break
                current_player = 3 - current_player  # Switch players
            else:
                print("That column is full. Try another one.")
        else:
            print("Invalid column. Please choose between 0 and 6.")

if __name__ == "__main__":
    play_game()

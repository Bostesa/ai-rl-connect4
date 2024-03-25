# utils.py

import numpy as np

def generate_hint(current_state, model):
    """
    Generates a hint for the player based on the current state.

    :param current_state: The current state of the game board.
    :param model: The trained model.
    :return: The best move as a hint.
    """
    observation = np.array(current_state).reshape((1, 6, 7))
    action, _ = model.predict(observation, deterministic=True)
    return int(action)

def undo_move(state_history):
    """
    Reverts to the previous game state.

    :param state_history: A list containing the history of states.
    :return: The previous game state.
    """
    if len(state_history) > 1:
        # Remove the last state (the current state) to revert to the previous state
        return state_history[-2]
    return state_history[0]  # Return the initial state if there's no previous state

def visualize_decision(current_state, model):
    """
    Generates visualization data for the AI's decision-making process.

    :param current_state: The current state of the game board.
    :param model: The trained model.
    :return: Visualization data.
    """
    # Example visualization: Show Q-values for each action
    # Note: This is a simplified example. The actual implementation will depend on your model and environment
    observation = np.array(current_state).reshape((1, 6, 7))
    q_values = model.predict(observation, deterministic=False)
    
    # Normalize Q-values for better visualization
    q_values_normalized = q_values - np.min(q_values)
    q_values_normalized /= np.max(q_values_normalized)
    
    visualization_data = {
        'q_values': q_values_normalized.tolist(),
        'recommended_action': np.argmax(q_values).tolist()
    }
    return visualization_data

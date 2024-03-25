from flask import Flask, request, jsonify
import numpy as np
from stable_baselines3 import Rainbow
from backend.RLagent import Connect4Env, SelfPlayUpdateCallback
from backend.utils import generate_hint, undo_move, visualize_decision

app = Flask(__name__)

# Load your trained model
model = Rainbow.load("path/to/connect4_rainbow_model.zip")
opponent_model = Rainbow.load("path/to/connect4_opponent_model.zip")

# Assuming you have a function to create your environment
env = Connect4Env()

@app.route('/make_move', methods=['POST'])
def make_move():
    data = request.json
    current_state = data['state']
    player_move = data.get('move', None)
    request_hint = data.get('request_hint', False)
    undo_request = data.get('undo', False)

    # Handle undo request
    if undo_request:
        new_state = undo_move(current_state)
        return jsonify({'state': new_state})

    # Handle move request
    if player_move is not None:
        # Update the environment with the player's move
        env.set_state(current_state)
        _, _, done, _ = env.step(player_move)
        if done:
            return jsonify({'state': env.get_state(), 'game_over': True})

    # AI makes its move
    observation = np.array(current_state).reshape((1, 6, 7))
    action, _ = model.predict(observation, deterministic=True)
    _, _, done, _ = env.step(action)

    # Handle hint request
    if request_hint:
        hint = generate_hint(current_state, opponent_model)

    # Generate visualization if required
    visualization_data = visualize_decision(current_state, model)

    response = {
        'action': int(action),
        'state': env.get_state(),
        'game_over': done,
        'visualization': visualization_data
    }

    if request_hint:
        response['hint'] = hint

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

# Terminal Chess

A terminal-based chess game with standard algebraic notation (SAN) input for 2 players.

## Features

- Terminal-based UI with Unicode chess pieces
- Command-based input using standard algebraic notation (SAN)
- Two-player mode
- Legal move validation
- Check and checkmate detection
- Move history display
- Undo functionality
- Save and load games
- Neural network-based engine with self-play learning

## Installation

This project uses `uv` for package management.

```bash
# Install uv if not already installed
pip install uv

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Usage

```bash
python main.py

# For self-play training
python self_play.py --games 100 --depth 2 --save-interval 10
```

## Controls

- Type algebraic notation (e.g., 'e4', 'Nf3', 'Bxc6', 'O-O') and press Enter to make a move
- Press 'u' to undo the last move
- Press 's' to save the game
- Press 'l' to load a saved game
- Press 'q' to quit

## Engine Training

The chess engine can be trained through self-play:

```bash
python self_play.py --games 100 --depth 2 --save-interval 10 --quiet
```

Options:
- `--games`: Number of games to play (default: 100)
- `--depth`: Search depth for the engine (default: 2)
- `--save-interval`: Save model every N games (default: 10)
- `--quiet`: Suppress detailed output
- `--model-name`: Custom name for the model file (default: auto-generated)

During training, you can:
- Press Ctrl+C to interrupt training and save the current model
- Type 'save' when prompted to save and exit

### Model Naming Scheme

Models are automatically named based on training parameters:

```
chess_g1000_d2_i10_nn_weights.npz
```

Where:
- `g1000`: Trained for 1000 games
- `d2`: Using search depth 2
- `i10`: Saved every 10 games

Checkpoint models are also saved during training:

```
checkpoint_100_chess_g1000_d2_i10_nn_weights.npz
```

This indicates a checkpoint after 100 games of a 1000-game training session.

The trained model will be automatically loaded when you play against the engine in the main game.

## Standard Algebraic Notation (SAN) Examples

- Pawn moves: `e4`, `d5`
- Knight moves: `Nf3`, `Nc6`
- Bishop moves: `Bc4`, `Bf5`
- Rook moves: `Ra3`, `Rd8`
- Queen moves: `Qh5`, `Qd7`
- King moves: `Ke2`, `Kf8`
- Captures: `Bxe5`, `Nxd4`
- Castling: `O-O` (kingside), `O-O-O` (queenside)
- Check: `Qh5+`
- Checkmate: `Qh7#`
- Promotion: `e8=Q`

## Requirements

- Python 3.7+
- Terminal with Unicode support

# Chess Engine with AlphaZero-style Training

This project implements a chess engine with a neural network evaluation function that can be trained on master-level games and refined with self-play.

## Features

- Neural network architecture similar to AlphaZero
- Training on Stockfish-generated dataset
- Self-play reinforcement learning
- Detailed metrics and visualizations
- Progressive training phases

## Installation

Clone the repository and install the required packages:

```bash
git clone [repository-url]
cd chess2
pip install -r requirements.txt
```

Required packages:
- python-chess
- numpy
- torch
- matplotlib
- tqdm
- datasets (for Hugging Face dataset loading)

## Training Pipeline

### Complete Training Workflow

For a complete training process (dataset → self-play → final model), use the `train_and_refine.py` script:

```bash
cd chess_terminal
python train_and_refine.py
```

This will:
1. Train on the LAION strategic chess dataset (500,000 samples)
2. Refine the model with self-play (200 games × 5 iterations)
3. Save the final model as `final_model.pt`

You can customize the training with various parameters:

```bash
python train_and_refine.py --dataset-samples 1000000 --dataset-epochs 30 \
                         --self-play-games 500 --self-play-iterations 10 \
                         --self-play-depth 6 --self-play-mcts 2000 \
                         --final-model my_strong_engine.pt
```

### Individual Training Components

#### Dataset Training Only

If you only want to train on the dataset:

```bash
python train.py --max-samples 500000 --epochs 20 --batch-size 512 \
               --output-model ../saved_models/my_trained_model.pt
```

#### Self-Play Training Only

If you only want to do self-play training:

```bash
python self_play.py --games 200 --iterations 5 --depth 5 --mcts-sims 1200 \
                   --batch-size 256 --model-name my_self_play_model.pt
```

Or to finetune from an existing model:

```bash
python self_play.py --games 200 --iterations 5 --depth 5 --mcts-sims 1200 \
                   --batch-size 256 --model-name refined_model.pt \
                   --finetune-from ../saved_models/base_model.pt
```

## Training Phases for Maximum Strength

For creating a tournament-level chess engine, we recommend a three-phase approach:

1. **Base Model Training**:
   ```bash
   python train.py --max-samples 1000000 --epochs 30 --batch-size 512 \
                  --output-model ../saved_models/base_model.pt
   ```

2. **Tactical Refinement**:
   ```bash
   python self_play.py --games 500 --iterations 10 --depth 4 --mcts-sims 800 \
                      --batch-size 512 --model-name refined_model.pt \
                      --finetune-from ../saved_models/base_model.pt
   ```

3. **Strategic Depth**:
   ```bash
   python self_play.py --games 200 --iterations 5 --depth 7 --mcts-sims 1600 \
                      --batch-size 512 --model-name final_model.pt \
                      --finetune-from ../saved_models/refined_model.pt
   ```

## Output and Metrics

The training process generates:

- **Trained Models**: Saved in the `saved_models` directory
- **Training Metrics**: Loss and accuracy plots
- **Game Statistics**: Win/loss/draw ratios, move counts
- **PGN Files**: Full game records for analysis

## Playing Against the Engine

Once trained, you can play against the engine using the terminal UI:

```bash
python main.py --model ../saved_models/final_model.pt
```

# Pearl Chess Engine

Pearl is a chess engine and AI training framework designed to learn from master games and improve through self-play.

## Engine Improvements

The Pearl chess engine has been enhanced with several classical chess engine techniques:

### 1. Attack Maps
- Added support for attack maps to improve tactical awareness
- Implemented functions to generate and evaluate attack patterns
- Enhanced position evaluation by considering piece attacks and threats

### 2. Move Pruning
- Implemented futility pruning to skip moves unlikely to improve the position
- Added late move reduction (LMR) to search promising moves more deeply
- Enhanced move ordering with MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)

### 3. Value Maps
- Added piece-square tables for different game phases
- Implemented position-dependent piece evaluation
- Enhanced evaluation by considering piece placement and coordination

### 4. Iterative Deepening
- Upgraded iterative deepening to allow for up to 3 minutes of thinking time
- Implemented time management to ensure efficient use of allocated time
- Added principal variation search for more efficient tree exploration

### 5. Transposition Tables
- Added support for efficient transposition tables with aging mechanism
- Implemented memory-efficient storage with LRU replacement strategy
- Enhanced search speed by reusing previously evaluated positions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pearl.git
cd pearl
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Playing Against the Engine

To play against the Pearl chess engine:

```bash
python main.py --mode play
```

### Training the Model

To train the model using the dataset of master games:

```bash
python main.py --mode train
```

### Converting PGN to CSV

To convert PGN files to the training dataset format:

```bash
python src/pgn_to_csv.py
```

This will process all PGN files in the `dataset/mastergames` directory and append the data to `dataset/dataset.csv`.

### Engine Self-Play

To have the engine play against itself to generate training data:

```bash
python main.py --mode selfplay --games 100
```

### Evaluating Positions

To evaluate a specific position:

```bash
python main.py --mode eval --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

## Project Structure

- `src/engine/`: Core chess engine components
  - `search.py`: Alpha-beta search with enhancements
  - `score.py`: Position evaluation functions
  - `movePick.py`: Move ordering for efficient search
  - `memory.py`: Transposition table implementation
- `src/modes/`: Different operation modes
  - `etrain.py`: Training functionality
  - `eplay.py`: Interactive play mode
  - `eselfplay.py`: Self-play for data generation
- `src/nnue/`: Neural network components
- `dataset/`: Training data
  - `mastergames/`: PGN files of master games
  - `dataset.csv`: Processed training data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Gesture-Snake-Game 🐍

A modern take on the classic Snake game that uses computer vision and hand gestures to control the snake's movement.

![image](https://github.com/user-attachments/assets/a6d96d13-27f3-42b7-a290-12a1c257da85)


## 🎮 Overview

This project demonstrates how to create an interactive game using Python libraries and computer vision. The snake is controlled by hand gestures captured through your webcam - simply swipe your index finger in the direction you want the snake to move!

This entire project was created with the assistance of Amazon Q, AWS's AI coding assistant, as part of the AWS Builders Challenge.

## ✨ Features

- **Gesture Controls**: Control the snake using hand movements
- **Responsive Design**: Immediate direction changes based on gestures
- **Visual Feedback**: Colorful trails follow your finger movements
- **Enhanced Graphics**: Vibrant snake with gradient coloring and animated eyes
- **User Interface**: Start screen, score display, and game over screen with restart option
- **Backup Controls**: Keyboard arrow keys as an alternative control method

## 🛠️ Technologies Used

- **pygame**: Game engine and graphics
- **OpenCV**: Computer vision and webcam integration
- **MediaPipe**: Hand tracking and gesture recognition
- **NumPy**: Mathematical operations and array handling

## 🚀 Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Gesture-Snake-Game.git
   cd Gesture-Snake-Game
   ```

2. Install the required dependencies:
   ```
   pip install pygame numpy opencv-python mediapipe
   ```

3. Run the game:
   ```
   python snake_game_improved.py
   ```

## 🎯 How to Play

1. Start the game and allow webcam access
2. Use your index finger to make swiping gestures:
   - Swipe RIGHT to move RIGHT
   - Swipe LEFT to move LEFT
   - Swipe UP to move UP
   - Swipe DOWN to move DOWN
3. Collect food (red apples) to grow your snake and increase your score
4. Avoid colliding with yourself
5. Use the on-screen buttons to start a new game or quit

## 🔧 Configuration

You can modify these variables in the code to adjust the game:
- `GRID_SIZE`: Change the size of the grid cells (larger = fewer squares)
- `FPS`: Adjust the frame rate
- `gesture_threshold`: Modify sensitivity of gesture detection
- `move_delay`: Change the snake's movement speed

## 🌟 Created with Amazon Q

This project was developed as part of the AWS Builders Challenge to showcase the capabilities of Amazon Q. The entire game was created by prompting Amazon Q with ideas and requirements, demonstrating how AI can help developers build complex applications with minimal manual coding.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Amazon Q for assistance in developing the entire codebase
- AWS Builders Challenge for the inspiration
- The pygame, OpenCV, and MediaPipe communities for their excellent libraries

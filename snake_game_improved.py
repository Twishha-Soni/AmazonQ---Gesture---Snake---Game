import pygame
import numpy as np
import cv2
import mediapipe as mp
import random
import time
import sys
import math

# Initialize pygame
pygame.init()

# Game constants
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 40  # Larger grid size means fewer squares
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
FPS = 12  # Reduced from 15 for slower gameplay

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
SNAKE_COLOR = (50, 205, 50)
FOOD_COLOR = (255, 50, 50)
BG_COLOR = (10, 20, 30)

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game with Fast Gesture Controls")
clock = pygame.time.Clock()

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Button class for UI elements
class Button:
    def __init__(self, x, y, width, height, text, color, hover_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.current_color = color
        self.font = pygame.font.SysFont('Arial', 24)
        self.text_surf = self.font.render(text, True, WHITE)
        self.text_rect = self.text_surf.get_rect(center=self.rect.center)
        
    def draw(self, surface):
        pygame.draw.rect(surface, self.current_color, self.rect, border_radius=10)
        pygame.draw.rect(surface, WHITE, self.rect, 2, border_radius=10)  # Border
        surface.blit(self.text_surf, self.text_rect)
        
    def is_hovered(self, pos):
        return self.rect.collidepoint(pos)
        
    def update(self, pos):
        if self.is_hovered(pos):
            self.current_color = self.hover_color
        else:
            self.current_color = self.color
            
    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.is_hovered(pos)
        return False

# Gesture tracking class for responsive swipe detection
class GestureTracker:
    def __init__(self):
        self.prev_positions = []
        self.max_positions = 4  # Reduced from 6 for even faster response
        self.gesture_threshold = 0.02  # Lower threshold for higher sensitivity
        self.direction_buffer = []
        self.buffer_size = 1  # Reduced to 1 for immediate direction changes
        self.last_significant_move_time = time.time()
        self.clear_interval = 0.5
        self.trail_colors = []
        self.last_direction = None
        self.direction_cooldown = 0.05  # Reduced cooldown for faster direction changes
        self.last_direction_time = 0
        
    def add_position(self, x, y):
        current_time = time.time()
        
        # If we have previous positions, check for movement
        if self.prev_positions:
            prev_x, prev_y = self.prev_positions[-1]
            distance = math.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            
            # If significant movement detected, update the time
            if distance > self.gesture_threshold:
                self.last_significant_move_time = current_time
                # Add a color for this trail segment
                hue = (current_time * 50) % 180
                color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                self.trail_colors.append((int(color[0]), int(color[1]), int(color[2])))
        
        # Add the new position
        self.prev_positions.append((x, y))
        if len(self.prev_positions) > self.max_positions:
            self.prev_positions.pop(0)
            if self.trail_colors:
                self.trail_colors.pop(0)
            
        # Clear trail if no significant movement for a while
        if current_time - self.last_significant_move_time > self.clear_interval:
            if len(self.prev_positions) > 1:
                self.prev_positions = [self.prev_positions[-1]]
                self.trail_colors = []
            self.direction_buffer = []
    
    def get_direction(self):
        if len(self.prev_positions) < 2:  # Need at least 2 points
            return None
        
        current_time = time.time()
        
        # If we're in cooldown period, don't change direction
        if current_time - self.last_direction_time < self.direction_cooldown:
            return self.last_direction
        
        # Just use the last two points for fastest response
        start_x, start_y = self.prev_positions[-2]
        end_x, end_y = self.prev_positions[-1]
        
        dx = end_x - start_x
        dy = end_y - start_y
        
        # Calculate distance moved
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Check if movement is significant enough
        if distance < self.gesture_threshold:
            return self.last_direction
        
        # Determine predominant direction with NORMAL horizontal controls
        if abs(dx) > abs(dy):
            new_dir = (1, 0) if dx > 0 else (-1, 0)  # Right swipe → Right movement, Left swipe → Left movement
        else:
            new_dir = (0, 1) if dy > 0 else (0, -1)  # Down or Up
        
        # With buffer size of 1, we immediately accept the new direction
        self.last_direction = new_dir
        self.last_direction_time = current_time
        return new_dir

# Initialize gesture tracker
gesture_tracker = GestureTracker()

# Snake class with improved responsiveness
class Snake:
    def __init__(self):
        self.positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.length = 1
        self.direction = (1, 0)  # Start moving right
        self.score = 0
        self.last_move_time = time.time()
        self.move_delay = 0.12  # Increased from 0.07 for slower movement
        self.next_direction = None  # Store next direction for more responsive controls
        
    def get_head_position(self):
        return self.positions[0]
    
    def update_direction(self, new_direction):
        # Prevent 180-degree turns
        if (new_direction[0] * -1, new_direction[1] * -1) == self.direction:
            return
            
        # Store the next direction to apply on the next move
        self.next_direction = new_direction
    
    def move(self):
        current_time = time.time()
        if current_time - self.last_move_time < self.move_delay:
            return False
            
        self.last_move_time = current_time
        
        # Apply the next_direction if it exists
        if self.next_direction:
            self.direction = self.next_direction
            self.next_direction = None
        
        head_x, head_y = self.get_head_position()
        dir_x, dir_y = self.direction
        new_x = (head_x + dir_x) % GRID_WIDTH
        new_y = (head_y + dir_y) % GRID_HEIGHT
        
        # Check for collision with self
        if (new_x, new_y) in self.positions[:-1]:
            return True  # Game over
            
        self.positions.insert(0, (new_x, new_y))
        if len(self.positions) > self.length:
            self.positions.pop()
            
        return False
    
    def draw(self, surface):
        for i, (x, y) in enumerate(self.positions):
            # Create a vibrant gradient effect
            progress = i / max(1, len(self.positions))
            
            # Use HSV color space for more vibrant colors
            hue = (120 - 60 * progress) % 180
            saturation = 255 - int(100 * progress)
            value = 255 - int(50 * progress)
            
            # Convert HSV to RGB
            hsv_color = np.uint8([[[hue, saturation, value]]])
            rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
            segment_color = (int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))
            
            # Draw snake segment with rounded corners
            rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            
            # Draw the main segment
            pygame.draw.rect(surface, segment_color, rect, border_radius=12)
            
            # Add eyes to the head
            if i == 0:
                # Determine eye positions based on direction
                eye_size = GRID_SIZE // 4
                
                if self.direction == (1, 0):  # Right
                    left_eye = (x * GRID_SIZE + GRID_SIZE * 3//4, y * GRID_SIZE + GRID_SIZE // 3)
                    right_eye = (x * GRID_SIZE + GRID_SIZE * 3//4, y * GRID_SIZE + GRID_SIZE * 2//3)
                elif self.direction == (-1, 0):  # Left
                    left_eye = (x * GRID_SIZE + GRID_SIZE // 4, y * GRID_SIZE + GRID_SIZE // 3)
                    right_eye = (x * GRID_SIZE + GRID_SIZE // 4, y * GRID_SIZE + GRID_SIZE * 2//3)
                elif self.direction == (0, -1):  # Up
                    left_eye = (x * GRID_SIZE + GRID_SIZE // 3, y * GRID_SIZE + GRID_SIZE // 4)
                    right_eye = (x * GRID_SIZE + GRID_SIZE * 2//3, y * GRID_SIZE + GRID_SIZE // 4)
                else:  # Down
                    left_eye = (x * GRID_SIZE + GRID_SIZE // 3, y * GRID_SIZE + GRID_SIZE * 3//4)
                    right_eye = (x * GRID_SIZE + GRID_SIZE * 2//3, y * GRID_SIZE + GRID_SIZE * 3//4)
                
                # Draw eyes
                pygame.draw.circle(surface, WHITE, left_eye, eye_size)
                pygame.draw.circle(surface, WHITE, right_eye, eye_size)
                pygame.draw.circle(surface, BLACK, left_eye, eye_size // 2)
                pygame.draw.circle(surface, BLACK, right_eye, eye_size // 2)
    
    def grow(self):
        self.length += 1
        self.score += 10
        # Speed up as snake grows, but more gradually
        self.move_delay = max(0.09, self.move_delay * 0.98)

# Food class
class Food:
    def __init__(self):
        self.position = (0, 0)
        self.randomize_position()
        
    def randomize_position(self):
        self.position = (random.randint(0, GRID_WIDTH - 1), 
                         random.randint(0, GRID_HEIGHT - 1))
    
    def draw(self, surface):
        x, y = self.position
        
        # Draw apple-like food
        rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.circle(surface, FOOD_COLOR, rect.center, GRID_SIZE // 2)
        
        # Add stem
        stem_color = (139, 69, 19)  # Brown
        stem_rect = pygame.Rect(
            rect.centerx - 2,
            rect.centery - GRID_SIZE // 2,
            4, GRID_SIZE // 4
        )
        pygame.draw.rect(surface, stem_color, stem_rect)
        
        # Add shine effect
        shine_pos = (rect.centerx - GRID_SIZE // 6, rect.centery - GRID_SIZE // 6)
        pygame.draw.circle(surface, (255, 200, 200), shine_pos, GRID_SIZE // 8)

# Function to process hand landmarks and determine direction
def process_hand_landmarks(landmarks):
    if not landmarks:
        return None
    
    # Get index finger tip coordinates
    index_finger_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Add position to tracker
    gesture_tracker.add_position(index_finger_tip.x, index_finger_tip.y)
    
    # Get direction from gesture
    return gesture_tracker.get_direction()

# Function to draw background grid
def draw_grid(surface):
    # Simple grid for better performance
    for y in range(0, HEIGHT, GRID_SIZE):
        for x in range(0, WIDTH, GRID_SIZE):
            rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(surface, (30, 30, 40), rect, 1)

# Function to display score with enhanced visuals
def display_score(surface, score):
    # Create score panel
    score_rect = pygame.Rect(10, 10, 150, 50)
    pygame.draw.rect(surface, (30, 50, 70), score_rect, border_radius=10)
    pygame.draw.rect(surface, (100, 150, 200), score_rect, 2, border_radius=10)  # Border
    
    # Display score text
    font = pygame.font.SysFont('Arial', 28)
    score_text = font.render(f'Score: {score}', True, WHITE)
    surface.blit(score_text, (score_rect.x + 20, score_rect.y + 12))

# Function to display start screen
def start_screen(surface):
    surface.fill(BG_COLOR)
    
    # Create title
    font_large = pygame.font.SysFont('Arial', 60)
    font_small = pygame.font.SysFont('Arial', 24)
    
    title_text = font_large.render('Snake Game', True, (100, 200, 100))
    subtitle_text = font_small.render('Control with hand gestures', True, WHITE)
    
    surface.blit(title_text, (WIDTH//2 - title_text.get_width()//2, HEIGHT//4))
    surface.blit(subtitle_text, (WIDTH//2 - subtitle_text.get_width()//2, HEIGHT//4 + 70))
    
    # Create start button
    start_button = Button(WIDTH//2 - 100, HEIGHT//2, 200, 60, 'Start Game', (50, 150, 50), (70, 200, 70))
    
    # Instructions
    instructions = [
        "How to play:",
        "- Swipe RIGHT to move RIGHT",
        "- Swipe LEFT to move LEFT",
        "- Swipe UP to move UP",
        "- Swipe DOWN to move DOWN",
        "",
        "- Collect food to grow",
        "- Avoid hitting yourself"
    ]
    
    instruction_font = pygame.font.SysFont('Arial', 20)
    for i, line in enumerate(instructions):
        text = instruction_font.render(line, True, (200, 200, 200))
        surface.blit(text, (WIDTH//2 - 150, HEIGHT//2 + 100 + i*25))
    
    pygame.display.update()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            mouse_pos = pygame.mouse.get_pos()
            start_button.update(mouse_pos)
            
            if start_button.is_clicked(mouse_pos, event):
                waiting = False
                
        start_button.draw(surface)
        pygame.display.update()
        clock.tick(30)

# Function to display game over screen with restart button
def game_over_screen(surface, score):
    surface.fill((20, 20, 30))  # Darker background
    
    # Create game over text
    font_large = pygame.font.SysFont('Arial', 60)
    font_medium = pygame.font.SysFont('Arial', 36)
    
    game_over_text = font_large.render('GAME OVER', True, RED)
    score_text = font_medium.render(f'Final Score: {score}', True, WHITE)
    
    surface.blit(game_over_text, (WIDTH//2 - game_over_text.get_width()//2, HEIGHT//4))
    surface.blit(score_text, (WIDTH//2 - score_text.get_width()//2, HEIGHT//4 + 80))
    
    # Create buttons
    restart_button = Button(WIDTH//2 - 220, HEIGHT//2 + 50, 200, 60, 'Play Again', (50, 100, 150), (70, 130, 200))
    quit_button = Button(WIDTH//2 + 20, HEIGHT//2 + 50, 200, 60, 'Quit Game', (150, 50, 50), (200, 70, 70))
    
    pygame.display.update()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            mouse_pos = pygame.mouse.get_pos()
            restart_button.update(mouse_pos)
            quit_button.update(mouse_pos)
            
            if restart_button.is_clicked(mouse_pos, event):
                return True  # Restart game
            elif quit_button.is_clicked(mouse_pos, event):
                pygame.quit()
                sys.exit()
                
        restart_button.draw(surface)
        quit_button.draw(surface)
        pygame.display.update()
        clock.tick(30)

# Main game function
def main():
    # Show start screen
    start_screen(screen)
    
    snake = Snake()
    food = Food()
    game_over = False
    
    # Game loop
    running = True
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # Manual controls as backup
                elif event.key == pygame.K_UP:
                    snake.update_direction((0, -1))
                elif event.key == pygame.K_DOWN:
                    snake.update_direction((0, 1))
                elif event.key == pygame.K_LEFT:
                    snake.update_direction((-1, 0))
                elif event.key == pygame.K_RIGHT:
                    snake.update_direction((1, 0))
        
        # Process webcam feed for hand tracking
        ret, frame = cap.read()
        if ret:
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and get hand landmarks
            results = hands.process(rgb_frame)
            
            # Draw hand landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Get direction from hand gesture
                    direction = process_hand_landmarks(hand_landmarks)
                    if direction:
                        snake.update_direction(direction)
                
                # Draw the finger trail
                if len(gesture_tracker.prev_positions) >= 2:
                    h, w, _ = frame.shape
                    for i in range(1, len(gesture_tracker.prev_positions)):
                        pt1 = (int(gesture_tracker.prev_positions[i-1][0] * w), 
                               int(gesture_tracker.prev_positions[i-1][1] * h))
                        pt2 = (int(gesture_tracker.prev_positions[i][0] * w), 
                               int(gesture_tracker.prev_positions[i][1] * h))
                        
                        # Gradient color based on recency
                        alpha = i / len(gesture_tracker.prev_positions)
                        color = (int(255 * (1-alpha)), int(255 * alpha), 255)
                        thickness = int(5 * alpha) + 1
                        
                        cv2.line(frame, pt1, pt2, color, thickness)
                
                # Draw direction indicator
                if direction:
                    h, w, _ = frame.shape
                    center = (w // 2, h - 50)
                    end_point = (center[0] + direction[0] * 50, center[1] + direction[1] * 50)
                    cv2.arrowedLine(frame, center, end_point, (0, 255, 255), 5)
                    
                    # Add text with direction name
                    dir_name = ""
                    if direction == (1, 0): dir_name = "RIGHT"
                    elif direction == (-1, 0): dir_name = "LEFT"
                    elif direction == (0, -1): dir_name = "UP"
                    elif direction == (0, 1): dir_name = "DOWN"
                    
                    cv2.putText(frame, dir_name, (10, h - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Hand Tracking', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
        
        if not game_over:
            # Move snake
            game_over = snake.move()
            
            # Check for food collision
            if snake.get_head_position() == food.position:
                snake.grow()
                food.randomize_position()
                # Make sure food doesn't appear on snake
                while food.position in snake.positions:
                    food.randomize_position()
            
            # Draw everything
            screen.fill(BG_COLOR)
            draw_grid(screen)
            snake.draw(screen)
            food.draw(screen)
            display_score(screen, snake.score)
            
            pygame.display.update()
            clock.tick(FPS)
        else:
            # Show game over screen and check if player wants to restart
            if game_over_screen(screen, snake.score):
                # Reset game
                snake = Snake()
                food = Food()
                game_over = False
            else:
                running = False
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()
